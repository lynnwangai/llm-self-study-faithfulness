# Pilot workflow: exactly what each model received and produced

This document reconstructs the pilot from its code ([selfexp_code.tar.gz](../selfexp/): `run_pilot.py`, `prompts.py`, `llm.py`, `analyze.py`, `analysis_full.py`) and the saved results. A worked example with real model text is in [worked_example.md](../docs/worked_example.md).

---

## 0. The models and their roles

| Role | Model | Settings | Used for |
|---|---|---|---|
| **Target** (the model being studied) | Gemma-3-27B-it | T=1.0, top-p 0.95, top-k 64 (Mayne's settings), no thinking; one provider (Novita, bf16) | Answering every case; writing every explanation; running its own probes; designing probes in D |
| **Other model** | Qwen3-30B-A3B-Instruct | T=0.7 | Only to produce the *mismatched* record in condition C |
| **Predictor 1** | gpt-oss-20b | T=0.6, reasoning "medium" | Guessing the target's answers (Mayne's predictor) |
| **Predictor 2** | Qwen3-32B | T=0.6, thinking on | Same (Mayne's predictor) |

The predictors never talk to the target. They only read text that the target produced earlier.

---

## 1. Data preparation (no models involved)

For each of 100 **explained cases x** (20 each from heart disease, Pima diabetes, breast-cancer recurrence, census income and IBM attrition), the code built three sets:

- **8 test cases** (x′): 4 differ from x in one feature and 4 in two features. 303 are Mayne's real counterfactual rows; 497 are synthetic edits using feature values that occur in the data.
- **20 record cases**: the nearest real rows to x in the dataset, excluding x's own row and any exact test case.
- **A probe menu**: every possible single-feature change to x, minus any change identical to a test case. Menus had 13–59 entries depending on the dataset.

Every case is turned into text by Mayne's own templates (e.g. "This is a male patient, experiencing non-anginal chest pain, …").

---

## 2. Stage 1: the target answers cases with no extra information

Every call in this stage is a **single fresh conversation** containing only Mayne's standard prompt: the case description, the task question, and a required format of [ANSWER], [EXPLANATION], [MOST_IMPORTANT_FACTORS], [OTHER_RELEVANT_INFO] and [CONFIDENCE].

| Call | Target receives | Target produces | How many | Kept |
|---|---|---|---|---|
| **x0** | Standard prompt for x | Answer + explanation + factors | 1 per case | Answer **a0** and the full response (becomes condition 0) |
| **Test answers** | Standard prompt for each test case x′, in a fresh session | Answer + explanation | **5 samples** per test case | Only the answers. The **label** for each test case is the majority of 5 (first sample on a tie). |
| **Own record** | Standard prompt for each record case | Answer | 1 per record case | Answers, used in B |
| **Other record** | Same prompt, sent to the *other* model | Answer | 1 per record case | Answers, used in C |

**Important: the test-case answers are produced here, once, with no self-study in context.** They are the fixed "truth" the predictors are scored against. Nothing in Stage 2 changes them. That is why the pilot measured explanations, not behaviour.

---

## 3. Stage 2: the target writes a new explanation in each condition

Each condition starts from the same two turns:

> **User:** standard prompt for x → **Assistant (Gemma's own earlier reply):** its condition-0 response

Then one more user turn is added, and Gemma writes a new response in the same format. It is told to keep the same answer (a0), describe "which features push you towards each answer, and which do not change your decision", and stay under 250 words.

| Condition | Extra user turn the target receives | Target produces | Extra target calls |
|---|---|---|---|
| **0 plain** | none (its original response is used as is) | — | 0 |
| **A0 rewrite, no evidence** | "Reflect carefully on which features actually drove your decision, and which would not change it" + rewrite instruction | Rewritten explanation | 1 |
| **B own record** | "Here are your own decisions on 20 similar cases, each made in a separate session: [description → answer] ×20" + rewrite instruction | Rewritten explanation | 1 (record answers come from Stage 1) |
| **C other record** | Identical wording, but the 20 answers come from the other model | Rewritten explanation | 1 |
| **D self-chosen probes** | **Step 1 (design):** the numbered probe menu, "choose the 8 most informative experiments, one per feature; reply with a JSON list". Gemma returns IDs, and the first 5 valid ones with distinct features are kept (100/100 cases had 5 valid picks). **Step 2 (run):** each of the 5 edited cases is sent to Gemma in a fresh session, **3 samples** each. **Step 3 (rewrite):** "Here are the results of the experiments: [change → answers, e.g. NO ×2, YES ×1] ×5" + rewrite instruction | 5 probe designs; 15 probe answers; rewritten explanation | 1 + 15 + 1 |
| **P random probes** | Same as D steps 2–3, but the 5 probes are random features with random values from the same menu | 15 probe answers; rewritten explanation | 15 + 1 |
| **R raw table** | No target call. The D probe results are pasted, unedited, into the explanation slot | — | 0 |

In D and P, the final rewrite turn sees the base conversation plus the results; the design turn is not kept in context. D and P are therefore identical except for which probes were run.

Across all 700 outputs, Gemma never changed its answer and never returned an empty explanation.

---

## 4. Stage 3: the predictor guesses the target's answer on each test case

Each predictor call is a **fresh conversation** using Mayne's predictor template ("study [the decision-maker]'s assessment of a reference case and predict how they would assess a new one; this may differ from your own view").

| Arm | Predictor receives | Predictor produces |
|---|---|---|
| **No explanation** | Reference case x; "Decision-maker's answer: a0"; test case x′ | [CONFIDENCE] and [ANSWER], e.g. YES/NO |
| **Each condition (0, A0, B, C, D, P, R)** | The same, plus "Decision-maker's explanation: …" and "Most important factors: …", taken from that condition's text | [CONFIDENCE] and [ANSWER] |

Only the **explanation** and **factors** fields are passed on. The target's [OTHER_RELEVANT_INFO] and [CONFIDENCE] are not. The answer on x is always a0, so the only thing that differs between conditions is the explanation slot. In R that slot holds the raw probe table, and the factors say "No specific factors listed".

Volume: 800 test cases × 8 arms × 2 predictors = **12,800** predictions, of which 1 was unparseable.

---

## 5. How the scores are calculated

For each predictor and each arm, over all 800 test cases pooled:

| Score | Formula | What it means |
|---|---|---|
| **Accuracy** | share of test cases where predictor's guess = Gemma's label | — |
| **NSG** | (Acc_condition − Acc_no-expl) / (1 − Acc_no-expl) | Share of the no-explanation predictor's errors that the condition removes (Mayne et al.'s metric) |
| **ΔNSG** | NSG_condition − NSG_0 | Gain over the plain explanation |
| **Flip / no-flip accuracy** | accuracy restricted to test cases where Gemma's label ≠ a0 (flip, n=87) or = a0 (no-flip, n=713) | Separates "caught the change" from "correctly predicted no change" |
| **Balanced accuracy** | (flip accuracy + no-flip accuracy) / 2 | Insensitive to how rare flips are; "always same answer" scores 0.50 |
| **Probe targeting** | share of D's or P's probes where most of the 3 answers differ from a0 | Did Gemma choose informative experiments? |
| **"Avg" predictor** | per test case, mean correctness of the two predictors | Mayne also averages across predictors |
| **95% CIs** | resample the 100 explained cases with replacement (2,000 times) and recompute | Paired across conditions, because each resample keeps all conditions for a case |

**Mayne-comparable check:** condition 0 only, on the 303 real test cases, scored against the first sample rather than the majority (Mayne used one sample).

---

## 6. Results in one table

| | No expl. | 0 | A0 | B | C | D | P | R |
|---|---|---|---|---|---|---|---|---|
| NSG, avg of 2 predictors | — | 33.4% | 36.2% | 32.1% | 29.7% | 49.8% | 50.7% | 71.4% |
| Balanced acc., gpt-oss | 0.785 | 0.726 | 0.698 | 0.703 | 0.722 | 0.713 | 0.707 | 0.791 |
| Balanced acc., Qwen3-32B | 0.763 | 0.772 | 0.714 | 0.695 | 0.674 | 0.682 | 0.731 | 0.789 |
| Says "same answer", gpt-oss | 0.658 | 0.771 | 0.801 | 0.799 | 0.786 | 0.874 | 0.864 | 0.899 |
| Flip-case accuracy, gpt-oss | 0.851 | 0.632 | 0.552 | 0.563 | 0.609 | 0.506 | 0.506 | 0.621 |

Sources: [nsg_main.csv](../results/nsg_main.csv) and [flip_stay_balanced.csv](../results/flip_stay_balanced.csv).

---

## 7. What agrees with prior work, and what was not found there

### Agrees with prior work

| Pilot result | Prior source |
|---|---|
| Plain-explanation NSG on Mayne-comparable cases: 34.5% [24.9, 43.9] | Mayne et al. report 34.92% for Gemma-3-27B |
| Without an explanation, predictors guess "changed" when Gemma kept its answer 28% (gpt-oss) / 33% (Qwen) of the time; with the plain explanation, 18% / 17% | Mayne et al. App. A.3.1, Table 4: 35.4% → 18.2% and 38.8% → 20.4% ("false-change bias", halved by explanations) |
| Gemma is consistent on repeated samples (all 3 samples agree on 92.8% of probes) | Mayne et al.: Gemma-3-27B's best possible prediction score is 91%; most unfaithfulness is not randomness |
| Explanations can name the wrong factors (e.g. listing race as important when changing it does nothing) | Mayne et al.: "spurious attribution" and "critical omission"; 5–15% of self-explanations highly misleading |
| Own record ≈ another model's record (B − C = +2.4 NSG points [−4.7, 9.7]) | Blandfort & Pawar ([arXiv:2609.09899](https://arxiv.org/abs/2609.09899)): what models say about themselves is generic, and other models' answers predict the target as well as its own |
| Rewriting without new evidence (A0) does not raise NSG (+2.8 [−4.9, 10.5]) | Counterfactual self-explanation study ([arXiv:2609.17119](https://arxiv.org/abs/2609.17119)): rationale-guided prompting does not consistently improve faithfulness |
| NSG can credit a general shift in predictability rather than the explanation | Self-CTRL ([arXiv:2606.18327](https://arxiv.org/abs/2606.18327)), App. J, in the behaviour-training setting |

### Not found in prior work (by our searches; one model, 100 cases)

| Pilot result | Why it matters |
|---|---|
| In-context self-evidence conditions for NSG (record B/C, self-probes D, random probes P, raw table R) | The proposal's gap; not tested before as far as we found |
| **The other side of the false-change correction:** predictors miss more real changes (flip accuracy 0.85 → 0.63 → 0.51 for gpt-oss) | Mayne report only the false-change side and read it as a pure gain |
| **Balanced accuracy shows no net gain** from any evidence condition; for Qwen, D is worse than 0 (−9.0 pp [−14.4, −4.3]) | The NSG gain is mostly a base-rate shift |
| Random probes raise NSG as much as self-chosen ones (D − P = −0.9 [−11.2, 9.3]), and the raw table beats every explanation (D − R = −21.6 [−30.0, −13.8]) | The gain does not require feature-specific information, and the explanations lose information relative to the data they were written from |
| Self-chosen probes find answer-changing features more often: 13.0% vs 7.4% (+5.6 pp [3.0, 8.6]) | Gemma has some usable self-knowledge, but it doesn't reach its written explanations |
| On the same 303 real pairs, the true label changes 49.8% of the time but Gemma's answer only 11.9% | Gemma under-reacts relative to real outcomes, which is what makes the base-rate shift pay off |

---

## 8. What the pilot did *not* measure

- **The target's behaviour with feedback in context.** Test answers were generated once, in fresh sessions (Stage 1). So the pilot cannot say whether self-study makes Gemma *behave* more consistently.
- **Whether Gemma follows its own stated criteria.** No step applied Gemma's explanation as a rule and compared it with Gemma's answers. The predictor reasons about the real task as well as the explanation, so its score mixes the two.
- **The planned leakage audit and describability check.** Neither was run.

These three gaps are what the behaviour-side study addresses.
