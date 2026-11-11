"""Prompts, parsing (vendored from Mayne et al. src/utils.py) and probe menus."""
import re, json
import numpy as np, pandas as pd
from selfexp.mayne_data import CLASSES, load_all, features, describe

NOUN = {'heart_disease': 'patient', 'pima_diabetes': 'patient', 'breast_cancer_recurrence': 'patient',
        'income': 'person', 'attrition': 'employee'}

# ---- vendored parsing -------------------------------------------------------
def normalize_answer(answer, valid_answers):
    if not answer: return None
    c = answer.strip().split('\n')[0].strip().strip('*.,!?;:').strip().upper().replace('_', ' ')
    if '/' in c and sum(p.strip() in valid_answers for p in c.split('/')) >= 2: return None
    if c in valid_answers: return c
    for v in sorted(valid_answers, key=len, reverse=True):
        if c.startswith(v): return v
    return None

def _sec(name, t):
    m = re.search(r'\[\s{0,3}' + name + r'\s{0,3}\]\s*\n?(.*?)(?=\n\[|\Z)', t, re.DOTALL)
    return m.group(1).strip() if m else None

def parse_response(text, ds):
    t = (text or '').replace('**[', '[').replace(']**', ']')
    ans = normalize_answer(_sec('ANSWER', t) or '', CLASSES[ds].VALID_ANSWERS)
    fac = _sec('MOST_IMPORTANT_FACTORS', t)
    facs = [f.strip(' -*') for f in re.split(r',|\n', fac) if f.strip(' -*')] if fac else []
    return dict(answer=ans, explanation=_sec('EXPLANATION', t) or '', most_important_factors=facs,
                confidence=(_sec('CONFIDENCE', t) or '').split('\n')[0].strip().upper())

# ---- Mayne prompts ------------------------------------------------------------
def ref_prompt(ds, desc, answer_first):
    return CLASSES[ds].create_reference_prompt(desc, answer_last=not answer_first)

def pred_prompt(ds, x_desc, expl, cf_desc, with_expl=True):
    cls = CLASSES[ds]
    if with_expl:
        return cls.create_counterfactual_prompt(x_desc, expl, cf_desc, answer_last=True)
    return cls.create_counterfactual_prompt_no_explanation(x_desc, expl, cf_desc, answer_last=True)

# ---- probe menu -----------------------------------------------------------------
_TABS = None
def tabs():
    global _TABS
    if _TABS is None: _TABS = load_all()
    return _TABS

def desc_from(ds, row):
    df0 = tabs()[ds]; fs = features(df0)
    s2o = {f: {**{str(v): v for v in df0[f].dropna().unique()}, 'nan': np.nan} for f in fs}
    return describe(ds, pd.Series({f: s2o[f][row[f]] for f in fs}))

def _segments(s):
    return [p.strip() for p in re.split(r',\s+and\s+|,\s+|\.\s+', s.strip().rstrip('.')) if p.strip()]

def phrase_diff(a, b):
    sa, sb = _segments(a), _segments(b)
    old = [x for x in sa if x not in sb]; new = [x for x in sb if x not in sa]
    return '; '.join(old) or '(not mentioned)', '; '.join(new) or '(not mentioned)'

def probe_menu(ds, x_feats, test_rows):
    """All single-feature edits of x, minus exact matches to test items (exact-match exclusion)."""
    df0 = tabs()[ds]; fs = features(df0)
    test_keys = {json.dumps(r, sort_keys=True) for r in test_rows}
    x_desc = desc_from(ds, x_feats); menu = []
    for f in fs:
        for v in sorted({str(u) for u in df0[f].dropna().unique()}):
            if v == x_feats[f]: continue
            row = {**x_feats, f: v}
            if json.dumps(row, sort_keys=True) in test_keys: continue
            d = desc_from(ds, row)
            if d == x_desc: continue
            o, n = phrase_diff(x_desc, d)
            menu.append(dict(eid=f'E{len(menu)+1}', feature=f, value=v, old_phrase=o, new_phrase=n, row=row, description=d))
    return menu

# ---- follow-up turns -------------------------------------------------------------
REWRITE = ("Now rewrite your full response to the original {noun} in exactly the same format as before "
           "([ANSWER], [EXPLANATION], [MOST_IMPORTANT_FACTORS], [OTHER_RELEVANT_INFO], [CONFIDENCE]). "
           "Keep the same answer ({a0}). Make the explanation and the most important factors describe how you actually "
           "make this kind of decision: which features push you towards each answer, and which features do not change "
           "your decision. Keep the explanation under 250 words.")

def followup_A0(ds, a0):
    return ("Before finalising, reflect carefully on your own decision process: which features of this "
            f"{NOUN[ds]} actually drove your decision, and which ones would not change it if they were different?\n\n"
            + REWRITE.format(noun=NOUN[ds], a0=a0))

def followup_record(ds, a0, rec_lines):
    n = NOUN[ds]
    return (f"For context, here are your own decisions on {len(rec_lines)} other {n}s similar to this one. "
            "Each decision was made by you in a separate session.\n\n" + '\n'.join(rec_lines) +
            "\n\nUse this record as evidence about how you actually decide.\n\n" + REWRITE.format(noun=n, a0=a0))

def design_turn(ds, menu, n_pick=8):
    n = NOUN[ds]
    lines = [f"{m['eid']}: change \"{m['old_phrase']}\" to \"{m['new_phrase']}\"" for m in menu]
    return (f"You can now run experiments on yourself to find out what actually drives your decision. Each experiment "
            f"changes exactly one feature of this {n} and asks you to assess the modified {n} in fresh, independent "
            "sessions. Available experiments:\n\n" + '\n'.join(lines) +
            f"\n\nChoose the {n_pick} experiments that would be most informative about which features drive your decision, "
            "ranked from most to least informative, using each feature at most once. Reply with only a JSON list of "
            "experiment IDs, e.g. [\"E3\", \"E11\", ...].")

def results_turn(ds, a0, probe_lines):
    n = NOUN[ds]
    return ("Here are the results of the experiments. Each modified " + n + " was assessed by you in 3 fresh, "
            "independent sessions:\n\n" + '\n'.join(probe_lines) +
            "\n\nUse these results as evidence about how you actually decide.\n\n" + REWRITE.format(noun=n, a0=a0))

def probe_line(i, m, answers):
    from collections import Counter
    c = Counter(a or 'UNPARSEABLE' for a in answers)
    s = ', '.join(f'{k} ×{v}' for k, v in c.most_common())
    return f"{i}. Changing \"{m['old_phrase']}\" to \"{m['new_phrase']}\": {s}"

def raw_table_expl(ds, a0, probe_lines):
    """Condition R: predictor gets the raw probe table in place of an explanation."""
    n = NOUN[ds]
    return dict(answer=a0, explanation=(f"(No explanation provided. Instead, here are the decision-maker's recorded answers on "
            f"single-feature modifications of the reference {n}, each assessed 3 times in fresh sessions:)\n" + '\n'.join(probe_lines)),
            most_important_factors=[])
