"""Async OpenRouter client with sqlite cache, cost ledger and spend caps."""
import os, json, hashlib, sqlite3, asyncio, time, random
import httpx

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_pilot', 'llm_cache.sqlite')
URL = 'https://openrouter.ai/api/v1/chat/completions'
HARD_CAP = float(os.environ.get('SELFEXP_HARD_CAP', 10.0))
SOFT_CAP = float(os.environ.get('SELFEXP_SOFT_CAP', 7.0))

# Decoding settings (Mayne et al. vllm_configs.py where applicable)
MODELS = {
    # target: Mayne-evaluated Gemma 3 27B (published NSG 34.92%), bf16, pinned to one provider for behavioural consistency
    'target':    dict(model='google/gemma-3-27b-it', temperature=1.0, top_p=0.95, top_k=64,
                      provider={'order': ['Novita'], 'allow_fallbacks': False}),
    'other':     dict(model='qwen/qwen3-30b-a3b-instruct-2507', temperature=0.7, top_p=0.8, top_k=20,
                      provider={'order': ['Nebius'], 'allow_fallbacks': False}),
    'pred_gptoss': dict(model='openai/gpt-oss-20b', temperature=0.6, reasoning={'effort': 'medium'},
                      provider={'order': ['DeepInfra'], 'allow_fallbacks': True}),
    'pred_qwen32': dict(model='qwen/qwen3-32b', temperature=0.6, top_p=0.95, top_k=20, reasoning={'enabled': True},
                      provider={'order': ['DeepInfra'], 'allow_fallbacks': True}),
}

# Routing overrides applied at send time only (cache key unchanged). Added mid-pilot when the DeepInfra
# Qwen3-32B endpoint degraded (30+ s latency, stalls); all Qwen3-32B providers serve fp8.
SEND_OVERRIDE = {'pred_qwen32': {'provider': {'order': ['DeepInfra', 'SiliconFlow'], 'ignore': ['Tenstorrent'], 'allow_fallbacks': True}}}

class BudgetExceeded(RuntimeError):
    pass

def _db():
    con = sqlite3.connect(DB, timeout=120)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('''CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, tag TEXT, model TEXT, request TEXT,
        text TEXT, reasoning TEXT, prompt_tokens INT, completion_tokens INT, reasoning_tokens INT,
        cost REAL, provider TEXT, finish TEXT, ts REAL)''')
    return con

def spent():
    con = _db(); v = con.execute('SELECT COALESCE(SUM(cost),0) FROM calls').fetchone()[0]; con.close(); return v

def ledger(group_by='tag'):
    con = _db()
    rows = con.execute(f'''SELECT {group_by}, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens), SUM(reasoning_tokens), SUM(cost)
        FROM calls GROUP BY {group_by} ORDER BY SUM(cost) DESC''').fetchall(); con.close(); return rows

class Client:
    def __init__(self, api_key, concurrency=24, allow_over_soft=False):
        self.key = api_key; self.sem = asyncio.Semaphore(concurrency)
        self.allow_over_soft = allow_over_soft
        self._spent = spent(); self.http = httpx.AsyncClient(timeout=httpx.Timeout(150, connect=20))

    def _check(self):
        if self._spent >= HARD_CAP:
            raise BudgetExceeded(f'hard cap ${HARD_CAP} reached (spent ${self._spent:.3f})')
        if self._spent >= SOFT_CAP and not self.allow_over_soft:
            raise BudgetExceeded(f'soft cap ${SOFT_CAP} reached (spent ${self._spent:.3f}); ask user')

    async def chat(self, role, messages, sample=0, tag='', max_tokens=2000, **over):
        cfg = {**MODELS[role], **over}
        body = {'messages': messages, 'max_tokens': max_tokens, 'usage': {'include': True}, **cfg}
        if 'seed' not in body: body['seed'] = 1000 + sample
        key = hashlib.sha256(json.dumps({**body, 'sample': sample}, sort_keys=True).encode()).hexdigest()
        send = {**body, **SEND_OVERRIDE.get(role, {})}   # routing-only changes that must not alter cache keys
        con = _db(); hit = con.execute('SELECT text, reasoning, finish FROM calls WHERE key=?', (key,)).fetchone(); con.close()
        if hit:
            return dict(text=hit[0], reasoning=hit[1], finish=hit[2], cached=True)
        async with self.sem:
            self._check()
            for attempt in range(6):
                try:
                    r = await asyncio.wait_for(self.http.post(URL, json=send, headers={'Authorization': f'Bearer {self.key}'}), timeout=240)
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                    r.raise_for_status(); j = r.json()
                    if 'choices' not in j or not j['choices']:
                        raise RuntimeError(f'no choices: {str(j)[:300]}')
                    if not (j['choices'][0].get('message', {}).get('content') or '').strip():
                        self._spent += float((j.get('usage') or {}).get('cost', 0) or 0)   # billed but empty: count, retry
                        raise RuntimeError('empty content')
                    break
                except (httpx.HTTPError, RuntimeError, json.JSONDecodeError, asyncio.TimeoutError) as e:
                    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 402: raise
                    if attempt == 5:
                        if str(e) == 'empty content' or isinstance(e, asyncio.TimeoutError):   # give up; do not cache
                            return dict(text='', reasoning='', finish='empty', cached=False)
                        raise
                    await asyncio.sleep(2 ** attempt + random.random())
        ch = j['choices'][0]; msg = ch.get('message', {}); u = j.get('usage', {}) or {}
        text = msg.get('content') or ''; reasoning = msg.get('reasoning') or ''
        rt = (u.get('completion_tokens_details') or {}).get('reasoning_tokens', 0) or 0
        cost = float(u.get('cost', 0) or 0); self._spent += cost
        con = _db()
        con.execute('INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', (key, tag, cfg['model'], json.dumps(body),
            text, reasoning, u.get('prompt_tokens', 0), u.get('completion_tokens', 0), rt, cost, j.get('provider', ''),
            ch.get('finish_reason', ''), time.time()))
        con.commit(); con.close()
        return dict(text=text, reasoning=reasoning, finish=ch.get('finish_reason'), cached=False)

    async def close(self):
        await self.http.aclose()
