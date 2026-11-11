"""Scoring: NSG with paired item-level bootstrap, cell breakdown, probe targeting."""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
D = 'data_pilot'

def labels(tb):
    out = {}
    for r in tb.itertuples():
        s = [a for a in json.loads(r.samples) if a]
        if not s: continue
        c = collections.Counter(s).most_common()
        modal = c[0][0] if (len(c) == 1 or c[0][1] > c[1][1]) else json.loads(r.samples)[0]
        out[r.test_id] = dict(first=json.loads(r.samples)[0], modal=modal, agree=c[0][1] / len(s))
    return out

def table(sfx=''):
    te = pd.read_parquet(f'{D}/testsets.parquet'); tb = pd.read_parquet(f'{D}/base_test{sfx}.parquet')
    pb = pd.read_parquet(f'{D}/predictions{sfx}.parquet'); cb = pd.read_parquet(f'{D}/conditions{sfx}.parquet')
    lab = labels(tb)
    pb = pb[pb.test_id.isin(lab)].copy()
    pb['label_modal'] = pb.test_id.map(lambda t: lab[t]['modal']); pb['label_first'] = pb.test_id.map(lambda t: lab[t]['first'])
    pb = pb.merge(te[['test_id', 'item_id', 'dataset', 'd', 'changed', 'real']], on='test_id')
    pb['correct_modal'] = (pb.pred == pb.label_modal).astype(float); pb['correct_first'] = (pb.pred == pb.label_first).astype(float)
    # probe-cell classification (by D's probes and by P's probes)
    probed = {}
    for r in cb[cb.cond.isin(['D', 'P'])].itertuples():
        probed[(r.item_id, r.cond)] = {p['feature'] for p in json.loads(r.meta)['probes']}
    def cell(row, which):
        pf = probed.get((row.item_id, which), set()); ch = set(json.loads(row.changed)); k = len(ch & pf)
        return f'd{row.d}_' + ({0: 'none', 1: 'one' if row.d == 2 else 'probed', 2: 'both'}[k]) + ('_probed' if False else '')
    pb['cell_D'] = [cell(r, 'D') for r in pb.itertuples()]; pb['cell_P'] = [cell(r, 'P') for r in pb.itertuples()]
    return pb, cb, tb, te

def nsg_from(acc_c, acc_0):
    return (acc_c - acc_0) / (1 - acc_0) if acc_0 < 1 else np.nan

def boot_nsg(pb, conds, label='modal', n_boot=2000, seed=0, subset=None):
    """Paired bootstrap over explained items. Returns rows per (predictor, cond)."""
    rng = np.random.default_rng(seed); col = f'correct_{label}'
    d = pb if subset is None else pb[subset(pb)]
    rows = []
    for pr, g in d.groupby('predictor'):
        piv = g.pivot_table(index='test_id', columns='cond', values=col, aggfunc='first')
        item = g.drop_duplicates('test_id').set_index('test_id').item_id.reindex(piv.index)
        items = item.unique(); idx = {i: np.where(item.values == i)[0] for i in items}
        def stats(rows_ix):
            sub = piv.iloc[rows_ix]; a0 = sub['noexp'].mean()
            return {c: nsg_from(sub[c].mean(), a0) for c in conds if c in sub}, {c: sub[c].mean() for c in ['noexp'] + conds if c in sub}
        full_n, full_a = stats(np.arange(len(piv)))
        bs = collections.defaultdict(list)
        for _ in range(n_boot):
            samp = rng.choice(items, len(items), replace=True)
            ix = np.concatenate([idx[i] for i in samp]); n, _a = stats(ix)
            for c in n: bs[c].append(n[c])
            for c in n:
                if c != '0' and '0' in n: bs[c + '-0'].append(n[c] - n['0'])
        for c in full_n:
            lo, hi = np.nanpercentile(bs[c], [2.5, 97.5])
            r = dict(predictor=pr, cond=c, n_tests=int(piv[c].notna().sum()), n_items=len(items), acc_noexp=full_a['noexp'],
                     acc=full_a[c], nsg=full_n[c], nsg_lo=lo, nsg_hi=hi)
            if c != '0' and '0' in full_n:
                dl = np.array(bs[c + '-0']); r.update(d_nsg=full_n[c] - full_n['0'], d_lo=np.nanpercentile(dl, 2.5),
                                                       d_hi=np.nanpercentile(dl, 97.5), p_le0=float(np.mean(dl <= 0)))
            rows.append(r)
    return pd.DataFrame(rows)

def targeting(cb, a0map):
    rows = []
    for r in cb[cb.cond.isin(['D', 'P'])].itertuples():
        m = json.loads(r.meta); pa = m['probe_answers']; a0 = a0map.get(r.item_id)
        for i, p in enumerate(m['probes']):
            s = [a for a in pa[i*3:(i+1)*3] if a]
            if not s: continue
            flip = np.mean([a != a0 for a in s])
            rows.append(dict(item_id=r.item_id, cond=r.cond, feature=p['feature'], flip_rate=flip, any_flip=float(flip > 0),
                             majority_flip=float(flip > 0.5), model_picked=(m.get('n_model_picked') or 0) > i if r.cond == 'D' else None))
    return pd.DataFrame(rows)

if __name__ == '__main__':
    sfx = sys.argv[1] if len(sys.argv) > 1 else ''
    pb, cb, tb, te = table(sfx)
    conds = [c for c in ['0', 'A0', 'B', 'C', 'D', 'P', 'R'] if c in set(pb.cond)]
    s = boot_nsg(pb, conds, n_boot=300); print(s.round(3).to_string())
    xb = pd.read_parquet(f'{D}/base_x{sfx}.parquet'); tg = targeting(cb, dict(zip(xb.item_id, xb.a0)))
    print(tg.groupby('cond')[['flip_rate', 'any_flip', 'majority_flip']].mean())
    print(pb[pb.cond.isin(['0', 'D', 'R'])].groupby(['cell_D', 'cond']).size().unstack())
