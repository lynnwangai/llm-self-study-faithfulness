"""Pilot runner. Stages: base -> conditions -> predict. All calls cached; re-running is free."""
import sys, os, json, asyncio, random, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from selfexp import prompts as P
from selfexp.llm import Client, BudgetExceeded, spent, ledger

D = 'data_pilot'
N_TEST_SAMPLES, N_PROBE_SAMPLES, N_PROBES = 5, 3, 5
CONDS = ['0', 'A0', 'B', 'C', 'D', 'P', 'R']
PREDICTORS = ['pred_gptoss', 'pred_qwen32']

def U(t): return {'role': 'user', 'content': t}
def A(t): return {'role': 'assistant', 'content': t}

def load():
    it = pd.read_parquet(f'{D}/items.parquet'); te = pd.read_parquet(f'{D}/testsets.parquet'); rc = pd.read_parquet(f'{D}/records.parquet')
    return it, te, rc

async def gather(coros):
    return await asyncio.gather(*coros)

# ----------------------------------------------------------------------------------
async def stage_base(cl, it, te, rc, limit=None):
    """Target: condition-0 response on x; 5 samples on each test item; 1 sample on records (target + other model)."""
    items = it if limit is None else it.groupby('dataset').head(limit)
    ids = set(items.item_id)
    async def ref(role, ds, desc, af, k, tag):
        r = await cl.chat(role, [U(P.ref_prompt(ds, desc, af))], sample=k, tag=tag, max_tokens=1500)
        return P.parse_response(r['text'], ds)['answer'], r['text']
    x_out = await gather([ref('target', r.dataset, r.description, r.answer_first, 0, 'x0') for r in items.itertuples()])
    af = dict(zip(it.item_id, it.answer_first))
    tt = te[te.item_id.isin(ids)]
    t_out = await gather([ref('target', r.dataset, r.description, af[r.item_id], k, 'test')
                          for r in tt.itertuples() for k in range(N_TEST_SAMPLES)])
    rr = rc[rc.item_id.isin(ids)]
    r_own = await gather([ref('target', r.dataset, r.description, af[r.item_id], 0, 'rec_own') for r in rr.itertuples()])
    r_oth = await gather([ref('other', r.dataset, r.description, af[r.item_id], 0, 'rec_other') for r in rr.itertuples()])
    xb = items[['item_id']].copy(); xb['a0'] = [a for a, _ in x_out]; xb['resp0'] = [t for _, t in x_out]
    tb = tt[['item_id', 'test_id']].copy()
    ans = [a for a, _ in t_out]
    tb['samples'] = [json.dumps(ans[i*N_TEST_SAMPLES:(i+1)*N_TEST_SAMPLES]) for i in range(len(tt))]
    rb = rr[['item_id', 'rec_id']].copy(); rb['own'] = [a for a, _ in r_own]; rb['other'] = [a for a, _ in r_oth]
    return xb, tb, rb

# ----------------------------------------------------------------------------------
async def stage_conditions(cl, it, te, rc, xb, rb, seed=0):
    rng = random.Random(seed)
    xb = xb.set_index('item_id'); out = []
    async def one(r):
        ds, iid = r.dataset, r.item_id
        if iid not in xb.index or xb.loc[iid, 'a0'] is None: return []
        a0, resp0 = xb.loc[iid, 'a0'], xb.loc[iid, 'resp0']
        base = [U(P.ref_prompt(ds, r.description, r.answer_first)), A(resp0)]
        res = [dict(item_id=iid, cond='0', expl=json.dumps(P.parse_response(resp0, ds)), meta='{}')]
        # A0: self-revision without new evidence
        t = await cl.chat('target', base + [U(P.followup_A0(ds, a0))], tag='cond_A0', max_tokens=1500)
        res.append(dict(item_id=iid, cond='A0', expl=json.dumps(P.parse_response(t['text'], ds)), meta='{}'))
        # B / C: record of own vs other model's answers on the same 20 kNN rows
        recs = rc[rc.item_id == iid].merge(rb, on=['item_id', 'rec_id'])
        for cond, col in (('B', 'own'), ('C', 'other')):
            lines = [f"{i+1}. {q.description} -> {getattr(q, col) or 'UNPARSEABLE'}" for i, q in enumerate(recs.itertuples())]
            t = await cl.chat('target', base + [U(P.followup_record(ds, a0, lines))], tag=f'cond_{cond}', max_tokens=1500)
            res.append(dict(item_id=iid, cond=cond, expl=json.dumps(P.parse_response(t['text'], ds)),
                            meta=json.dumps(dict(n_rec=len(lines), rec_answers=list(recs[col])))))
        # D (self-chosen probes) and P (random probes)
        test_rows = [json.loads(f) for f in te[te.item_id == iid].features]
        menu = P.probe_menu(ds, json.loads(r.features), test_rows); by_id = {m['eid']: m for m in menu}
        dt = await cl.chat('target', base + [U(P.design_turn(ds, menu))], tag='design_D', max_tokens=400)
        import re
        picked, feats = [], set()
        for eid in re.findall(r'E\d+', dt['text']):
            m = by_id.get(eid)
            if m and m['feature'] not in feats: picked.append(m); feats.add(m['feature'])
            if len(picked) == N_PROBES: break
        n_model_picked = len(picked)
        fill = [m for m in menu if m['feature'] not in feats]; rng.shuffle(fill)   # top up if model under-delivers
        for m in fill:
            if len(picked) == N_PROBES: break
            if m['feature'] not in feats: picked.append(m); feats.add(m['feature'])
        allf = list({m['feature'] for m in menu}); rng.shuffle(allf); rand = []
        for f in allf[:N_PROBES]:
            rand.append(rng.choice([m for m in menu if m['feature'] == f]))
        for cond, probes in (('D', picked), ('P', rand)):
            pans = await gather([cl.chat('target', [U(P.ref_prompt(ds, m['description'], r.answer_first))], sample=k,
                                         tag=f'probe', max_tokens=1500) for m in probes for k in range(N_PROBE_SAMPLES)])
            pans = [P.parse_response(p['text'], ds)['answer'] for p in pans]
            lines = [P.probe_line(i+1, m, pans[i*N_PROBE_SAMPLES:(i+1)*N_PROBE_SAMPLES]) for i, m in enumerate(probes)]
            meta = dict(probes=[dict(feature=m['feature'], value=m['value'], row=m['row']) for m in probes],
                        probe_answers=pans, n_model_picked=n_model_picked if cond == 'D' else None, design_raw=dt['text'] if cond == 'D' else None)
            if cond == 'D':
                res.append(dict(item_id=iid, cond='R', expl=json.dumps(P.raw_table_expl(ds, a0, lines)), meta=json.dumps(meta)))
            # D and P share an identical final turn; they differ only in which probes were run
            t = await cl.chat('target', base + [U(P.results_turn(ds, a0, lines))], tag=f'cond_{cond}', max_tokens=1500)
            res.append(dict(item_id=iid, cond=cond, expl=json.dumps(P.parse_response(t['text'], ds)), meta=json.dumps(meta)))
        return res
    for rs in await gather([one(r) for r in it.itertuples()]):
        out += rs
    return pd.DataFrame(out)

# ----------------------------------------------------------------------------------
async def stage_predict(cl, it, te, xb, cb, predictors=PREDICTORS, conds=CONDS):
    itx = it.set_index('item_id'); xa = dict(zip(xb.item_id, xb.a0))
    ex = {(r.item_id, r.cond): json.loads(r.expl) for r in cb.itertuples()}
    jobs, keys = [], []
    for t in te.itertuples():
        iid = t.item_id
        if iid not in xa or xa[iid] is None: continue
        ds = t.dataset; x = itx.loc[iid]
        for pr in predictors:
            # no-explanation baseline
            jobs.append(cl.chat(pr, [U(P.pred_prompt(ds, x.description, {'answer': xa[iid]}, t.description, with_expl=False))],
                                tag=f'pred_noexp', max_tokens=4000)); keys.append((t.test_id, pr, 'noexp'))
            for c in conds:
                e = ex.get((iid, c))
                if e is None: continue
                e = {**e, 'answer': xa[iid]}   # fixed predictor input: same reference answer in every condition
                jobs.append(cl.chat(pr, [U(P.pred_prompt(ds, x.description, e, t.description, with_expl=True))],
                                    tag=f'pred_{c}', max_tokens=4000)); keys.append((t.test_id, pr, c))
    res = await gather(jobs)
    ds_of = dict(zip(te.test_id, te.dataset))
    rows = [dict(test_id=k[0], predictor=k[1], cond=k[2], pred=P.parse_response(r['text'], ds_of[k[0]])['answer'],
                 conf=P.parse_response(r['text'], ds_of[k[0]])['confidence'], finish=r['finish']) for k, r in zip(keys, res)]
    return pd.DataFrame(rows)

async def main(args):
    if args.mock:
        from selfexp.mock import MockClient; cl = MockClient()
    else:
        cl = Client(os.environ[args.key_env], concurrency=args.conc, allow_over_soft=args.allow_over_soft)
    it, te, rc = load()
    if args.limit:
        it = it.groupby('dataset').head(args.limit); te = te[te.item_id.isin(it.item_id)]; rc = rc[rc.item_id.isin(it.item_id)]
    sfx = args.suffix
    try:
        if 'base' in args.stages:
            xb, tb, rb = await stage_base(cl, it, te, rc)
            xb.to_parquet(f'{D}/base_x{sfx}.parquet'); tb.to_parquet(f'{D}/base_test{sfx}.parquet'); rb.to_parquet(f'{D}/base_rec{sfx}.parquet')
        xb = pd.read_parquet(f'{D}/base_x{sfx}.parquet'); rb = pd.read_parquet(f'{D}/base_rec{sfx}.parquet')
        if 'cond' in args.stages:
            cb = await stage_conditions(cl, it, te, rc, xb, rb); cb.to_parquet(f'{D}/conditions{sfx}.parquet')
        if 'predict' in args.stages:
            cb = pd.read_parquet(f'{D}/conditions{sfx}.parquet')
            pb = await stage_predict(cl, it, te, xb, cb, predictors=args.predictors.split(','), conds=args.conds.split(','))
            pb.to_parquet(f'{D}/predictions{sfx}.parquet')
    except BudgetExceeded as e:
        print('BUDGET STOP:', e)
    finally:
        await cl.close()
    if not args.mock:
        print(f'spent ${spent():.4f}')
        for row in ledger(): print(row)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', default='base,cond,predict'); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--mock', action='store_true'); ap.add_argument('--suffix', default='')
    ap.add_argument('--key-env', default='OPENROUTER_API_KEY'); ap.add_argument('--conc', type=int, default=24)
    ap.add_argument('--allow-over-soft', action='store_true')
    ap.add_argument('--predictors', default=','.join(PREDICTORS)); ap.add_argument('--conds', default=','.join(CONDS))
    asyncio.run(main(ap.parse_args()))
