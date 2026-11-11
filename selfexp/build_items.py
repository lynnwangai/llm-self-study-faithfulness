"""Build pilot items: explained items x, test sets C(x), and k-NN record rows."""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pyarrow.parquet as pq
from selfexp.mayne_data import load_all, features, describe, reference_prompt

DATASETS = ['heart_disease', 'pima_diabetes', 'breast_cancer_recurrence', 'income', 'attrition']
N_PER_DS, N_TEST, N_D1, K_REC, SEED = 20, 8, 4, 20, 0
OUT = 'data_pilot'

def ham(a, b, feats):
    return sum(str(a[f]) != str(b[f]) for f in feats)

def main():
    rng = random.Random(SEED)
    tabs = load_all()
    comb = pq.read_table('mayne/data/natural_counterfactuals/combined_dataset.parquet',
        columns=['original_dataset','original_question_idx','original_answer_first','counterfactual_question_idx','counterfactual_hamming_distance']).to_pandas()
    items, tests, recs = [], [], []
    for ds in DATASETS:
        df0 = tabs[ds]; df = df0.copy(); feats = features(df)
        for f in feats: df[f] = df[f].map(str).astype(object)
        s2o = {f: {**{str(v): v for v in df0[f].dropna().unique()}, 'nan': np.nan} for f in feats}
        def desc(row):  # row: dict of str values -> description using original dtypes
            return describe(ds, pd.Series({f: s2o[f][row[f]] for f in feats}))
        vals = {f: sorted(v for v in df[f].unique() if v != 'nan') for f in feats}
        sub = comb[comb.original_dataset == ds]
        origs = sorted(sub.original_question_idx.astype(int).unique())
        chosen = rng.sample(origs, N_PER_DS)
        for xi in chosen:
            x = df.loc[xi]; item_id = f'{ds}:{xi}'
            af = bool(sub[sub.original_question_idx.astype(int) == xi].original_answer_first.iloc[0])
            items.append(dict(item_id=item_id, dataset=ds, row_idx=int(xi), answer_first=af,
                features=json.dumps({f: x[f] for f in feats}), description=desc({f: x[f] for f in feats}),
                ground_truth=int(df.loc[xi, 'target']), n_features=len(feats)))
            # real CFs from Mayne pairs
            cand = []
            for ci in sub[sub.original_question_idx.astype(int) == xi].counterfactual_question_idx.astype(int).unique():
                row = df.loc[ci]; d = ham(x, row, feats)
                if d in (1, 2): cand.append(dict(row={f: row[f] for f in feats}, real=True, src_idx=int(ci)))
            seen = {json.dumps(c['row'], sort_keys=True) for c in cand}
            seen.add(json.dumps({f: x[f] for f in feats}, sort_keys=True))
            d1 = [c for c in cand if ham(x, c['row'], feats) == 1]; d2 = [c for c in cand if ham(x, c['row'], feats) == 2]
            rng.shuffle(d1); rng.shuffle(d2)
            d1, d2 = d1[:N_D1], d2[:N_TEST - N_D1]
            def synth(d):
                for _ in range(200):
                    fs = rng.sample(feats, d); row = {f: x[f] for f in feats}
                    ok = True
                    for f in fs:
                        alt = [v for v in vals[f] if v != x[f]]
                        if not alt: ok = False; break
                        row[f] = rng.choice(alt)
                    key = json.dumps(row, sort_keys=True)
                    if ok and key not in seen:
                        seen.add(key); return dict(row=row, real=False, src_idx=-1)
                raise RuntimeError('synth failed')
            while len(d1) < N_D1: d1.append(synth(1))
            while len(d2) < N_TEST - N_D1: d2.append(synth(2))
            for j, c in enumerate(d1 + d2):
                changed = [f for f in feats if c['row'][f] != x[f]]
                tests.append(dict(item_id=item_id, test_id=f'{item_id}:t{j}', dataset=ds, d=len(changed),
                    changed=json.dumps(changed), real=c['real'], src_idx=c['src_idx'],
                    features=json.dumps(c['row']), description=desc(c['row'])))
            # k-NN record rows (real rows), excluding x and exact test matches
            test_keys = {json.dumps(c['row'], sort_keys=True) for c in d1 + d2}
            others = df.drop(index=xi)
            dist = (others[feats] != x[feats]).sum(axis=1)
            tie = pd.Series([rng.random() for _ in range(len(others))], index=others.index)
            order = pd.DataFrame({'d': dist, 't': tie}).sort_values(['d', 't']).index
            n = 0; seen_rec = set()
            for ri in order:
                row = {f: others.loc[ri, f] for f in feats}; key = json.dumps(row, sort_keys=True)
                if key in test_keys or key in seen_rec: continue
                seen_rec.add(key)
                recs.append(dict(item_id=item_id, rec_id=f'{item_id}:r{n}', dataset=ds, src_idx=int(ri), d=int(dist[ri]),
                    features=json.dumps(row), description=desc(row)))
                n += 1
                if n == K_REC: break
    items, tests, recs = map(pd.DataFrame, (items, tests, recs))
    items.to_parquet(f'{OUT}/items.parquet'); tests.to_parquet(f'{OUT}/testsets.parquet'); recs.to_parquet(f'{OUT}/records.parquet')
    summ = tests.groupby(['dataset', 'd', 'real']).size().unstack(['d', 'real'], fill_value=0)
    summ.columns = [f'd{d}_{"real" if r else "synth"}' for d, r in summ.columns]
    summ['n_items'] = items.groupby('dataset').size(); summ['n_features'] = items.groupby('dataset').n_features.first()
    summ['rec_mean_dist'] = recs.groupby('dataset').d.mean().round(2)
    summ.to_csv(f'{OUT}/data_summary.csv'); print(summ.to_string())
    print('items', len(items), 'tests', len(tests), 'records', len(recs))

if __name__ == '__main__':
    main()
