"""Full pilot analysis -> CSV tables used by the report and figures."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, collections
from selfexp.analyze import table, boot_nsg, targeting, labels
from selfexp.mayne_data import load_all, CLASSES
D = 'data_pilot'; OUT = 'results_pilot'; os.makedirs(OUT, exist_ok=True)
CONDS = ['0', 'A0', 'B', 'C', 'D', 'P', 'R']
NB = 2000

def avg_predictors(pb):
    """Mayne-style: average correctness over predictors per (test_id, cond) -> pseudo-predictor 'avg'."""
    g = pb.groupby(['test_id', 'cond', 'item_id'], as_index=False)[['correct_modal', 'correct_first']].mean()
    g['predictor'] = 'avg'; return g

def n_req(n, se, d):
    return int(np.ceil(n * (2.80 * se / abs(d)) ** 2)) if (np.isfinite(se) and np.isfinite(d) and d != 0) else None

def run(sfx=''):
    pb, cb, tb, te = table(sfx)
    xb = pd.read_parquet(f'{D}/base_x{sfx}.parquet'); a0 = dict(zip(xb.item_id, xb.a0))
    tabs = load_all()
    # ground truth for real test items
    gt = {}
    for r in te[te.real].itertuples():
        gt[r.test_id] = CLASSES[r.dataset].format_target(int(tabs[r.dataset].loc[r.src_idx, 'target']))
    pb['gt_label'] = pb.test_id.map(gt)
    both = pd.concat([pb, avg_predictors(pb).merge(pb[['test_id', 'cell_D', 'cell_P', 'real', 'gt_label', 'label_modal', 'label_first', 'd', 'dataset']].drop_duplicates('test_id'), on='test_id')])
    res = {}
    res['main'] = boot_nsg(both, CONDS, n_boot=NB)
    res['main_first_label'] = boot_nsg(both, CONDS, label='first', n_boot=NB)
    res['unprobed_by_D'] = boot_nsg(both, CONDS, n_boot=NB, subset=lambda d: d.cell_D.isin(['d1_none', 'd2_none']))
    res['probed_by_D'] = boot_nsg(both, CONDS, n_boot=NB, subset=lambda d: ~d.cell_D.isin(['d1_none', 'd2_none']))
    res['mayne_comparable'] = boot_nsg(both, ['0'], label='first', n_boot=NB, subset=lambda d: d.real)
    res['real_model_disagrees_gt'] = boot_nsg(both, CONDS, n_boot=NB, subset=lambda d: d.real & (d.gt_label != d.label_modal))
    cells = []
    for c in ['d1_none', 'd1_probed', 'd2_none', 'd2_one', 'd2_both']:
        s = boot_nsg(both, ['0', 'A0', 'D', 'R', 'P'], n_boot=500, subset=lambda d, c=c: d.cell_D == c); s['cell'] = c; cells.append(s)
    res['cells_by_D'] = pd.concat(cells)
    per_ds = []
    for ds in sorted(te.dataset.unique()):
        s = boot_nsg(both, CONDS, n_boot=500, subset=lambda d, ds=ds: d.dataset == ds); s['dataset'] = ds; per_ds.append(s)
    res['per_dataset'] = pd.concat(per_ds)
    for k, v in res.items(): v.to_csv(f'{OUT}/nsg_{k}.csv', index=False)
    # targeting (probe flip rates), item-level bootstrap of D - P
    tg = targeting(cb, a0); tg.to_csv(f'{OUT}/probe_targeting_raw.csv', index=False)
    it_fl = tg.groupby(['item_id', 'cond']).majority_flip.mean().unstack()
    rng = np.random.default_rng(0); diffs = []
    for _ in range(NB):
        s = it_fl.sample(len(it_fl), replace=True, random_state=rng.integers(1e9)); diffs.append(s.D.mean() - s.P.mean())
    tsum = pd.DataFrame([dict(flip_D=it_fl.D.mean(), flip_P=it_fl.P.mean(), diff=it_fl.D.mean() - it_fl.P.mean(),
                              lo=np.percentile(diffs, 2.5), hi=np.percentile(diffs, 97.5), n_items=len(it_fl))])
    tsum.to_csv(f'{OUT}/probe_targeting_summary.csv', index=False)
    # explanation lengths and answer-change rates
    ex = cb.expl.map(json.loads)
    ln = pd.DataFrame(dict(cond=cb.cond, words=ex.map(lambda e: len(e['explanation'].split())),
                           n_factors=ex.map(lambda e: len(e.get('most_important_factors', []))),
                           answer_changed=[(e['answer'] != a0.get(i)) for e, i in zip(ex, cb.item_id)]))
    ln.groupby('cond').agg(words_mean=('words', 'mean'), words_median=('words', 'median'), factors=('n_factors', 'mean'),
                           answer_changed=('answer_changed', 'mean')).to_csv(f'{OUT}/explanation_lengths.csv')
    # shortcut check on real test items
    real = pb[pb.real & pb.gt_label.notna()].drop_duplicates('test_id')
    sc = dict(n_real=len(real), gt_predicts_model=(real.gt_label == real.label_modal).mean(),
              target_label_agreement_5samples=np.mean([labels(tb)[t]['agree'] for t in real.test_id]))
    for pr, g in pb[pb.real & pb.gt_label.notna() & (pb.cond == 'noexp')].groupby('predictor'):
        sc[f'{pr}_noexp_acc_vs_model'] = (g.pred == g.label_modal).mean(); sc[f'{pr}_noexp_acc_vs_gt'] = (g.pred == g.gt_label).mean()
    pd.DataFrame([sc]).to_csv(f'{OUT}/shortcut_check.csv', index=False)
    # power for main contrasts (avg predictor)
    m = res['main']; m = m[m.predictor == 'avg']; rows = []
    for c in ['A0', 'B', 'C', 'D', 'P', 'R']:
        r = m[m.cond == c].iloc[0]; se = (r.d_hi - r.d_lo) / (2 * 1.96)
        rows.append(dict(contrast=f'{c}-0', d_nsg=r.d_nsg, se=se, n_items=r.n_items,
                         items_for_80pct_power=n_req(r.n_items, se, r.d_nsg)))
    # D - A0 contrast (format-matched)
    piv = both[both.predictor == 'avg'].pivot_table(index='test_id', columns='cond', values='correct_modal')
    item = both[both.predictor == 'avg'].drop_duplicates('test_id').set_index('test_id').item_id.reindex(piv.index)
    items = item.unique(); idx = {i: np.where(item.values == i)[0] for i in items}
    def nsg(sub, c): a = sub['noexp'].mean(); return (sub[c].mean() - a) / (1 - a)
    for c1, c2 in [('D', 'A0'), ('B', 'A0'), ('D', 'P'), ('D', 'R'), ('B', 'C'), ('D', 'B')]:
        full = nsg(piv, c1) - nsg(piv, c2); bs = []
        for _ in range(NB):
            ix = np.concatenate([idx[i] for i in rng.choice(items, len(items))]); sub = piv.iloc[ix]; bs.append(nsg(sub, c1) - nsg(sub, c2))
        se = np.nanstd(bs)
        rows.append(dict(contrast=f'{c1}-{c2}', d_nsg=full, se=se, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5), n_items=len(items),
                         items_for_80pct_power=n_req(len(items), se, full)))
    pd.DataFrame(rows).to_csv(f'{OUT}/contrasts_power.csv', index=False)
    return res

if __name__ == '__main__':
    r = run(sys.argv[1] if len(sys.argv) > 1 else '')
    m = r['main']; print(m[m.predictor == 'avg'].round(3).to_string())
    print(pd.read_csv(f'{OUT}/contrasts_power.csv').round(3).to_string())

def flip_stay(sfx='', n_boot=2000, seed=0):
    """Accuracy split by whether the target's label differs from its reference answer (flip) or not (stay).
    Balanced accuracy = mean(acc_flip, acc_stay). Includes the trivial 'always same as reference' predictor."""
    pb, cb, tb, te = table(sfx)
    xb = pd.read_parquet(f'{D}/base_x{sfx}.parquet'); a0 = dict(zip(xb.item_id, xb.a0))
    pb['flip'] = pb.item_id.map(a0) != pb.label_modal
    same = pb[pb.cond == 'noexp'].copy(); same['cond'] = 'always_same'; same['pred'] = same.item_id.map(a0)
    same['correct_modal'] = (same.pred == same.label_modal).astype(float)
    pb = pd.concat([pb, same])
    rng = np.random.default_rng(seed); rows = []
    for pr, g in pb.groupby('predictor'):
        items = g.item_id.unique()
        by_item = {i: d for i, d in g.groupby('item_id')}
        def stats(sub):
            out = {}
            for c, d in sub.groupby('cond'):
                f, s = d[d.flip].correct_modal.mean(), d[~d.flip].correct_modal.mean()
                out[c] = dict(acc=d.correct_modal.mean(), acc_flip=f, acc_stay=s, bal_acc=np.nanmean([f, s]), say_same=(d.pred == d.item_id.map(a0)).mean())
            return out
        full = stats(g); bs = collections.defaultdict(list)
        for _ in range(n_boot):
            sub = pd.concat([by_item[i] for i in rng.choice(items, len(items))]); st = stats(sub)
            for c in st:
                for k in st[c]: bs[(c, k)].append(st[c][k])
                if '0' in st: bs[(c, 'bal_minus_0')].append(st[c]['bal_acc'] - st['0']['bal_acc'])
        for c, v in full.items():
            r = dict(predictor=pr, cond=c, n_flip=int(g[(g.cond == c) & g.flip].shape[0]), n_stay=int(g[(g.cond == c) & ~g.flip].shape[0]), **v)
            for k in ['acc_flip', 'acc_stay', 'bal_acc']:
                r[k + '_lo'], r[k + '_hi'] = np.nanpercentile(bs[(c, k)], [2.5, 97.5])
            d0 = np.array(bs[(c, 'bal_minus_0')]); r['bal_minus_0'] = v['bal_acc'] - full['0']['bal_acc']
            r['bal_minus_0_lo'], r['bal_minus_0_hi'] = np.nanpercentile(d0, [2.5, 97.5])
            rows.append(r)
    out = pd.DataFrame(rows); out.to_csv(f'{OUT}/flip_stay_balanced.csv', index=False); return out
