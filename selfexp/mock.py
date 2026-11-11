"""Offline mock for plumbing tests: a fake rule-based 'model'."""
import random, re
class MockClient:
    async def chat(self, role, messages, sample=0, tag='', max_tokens=0, **kw):
        last = messages[-1]['content']; rnd = random.Random(hash((last[-200:], sample, role)))
        opts = re.findall(r'\[ANSWER\]\n(.*?)\(you must', last)
        o = (opts[0] if opts else 'YES or NO').split(' or ')
        a = rnd.choice([x.strip() for x in o])
        if 'JSON list of experiment IDs' in last:
            ids = re.findall(r'(E\d+):', last); rnd.shuffle(ids); return dict(text=str(ids[:8]), reasoning='', finish='stop', cached=False)
        return dict(text=f"[ANSWER]\n{a}\n\n[EXPLANATION]\nmock reasoning\n\n[MOST_IMPORTANT_FACTORS]\nage, sex\n\n[CONFIDENCE]\nHIGH", reasoning='', finish='stop', cached=False)
    async def close(self): pass
