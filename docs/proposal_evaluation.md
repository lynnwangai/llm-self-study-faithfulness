# Evaluation: "Recovering Self-Consistency Gains at Inference Time"

Evaluated 2026-10-24 against the cited papers, an arXiv scan (27 papers since 2025-06 on simulatability / self-explanation / self-prediction), and the released Mayne et al. benchmark (github.com/HarryMayne/faithfulness).

## 1. Verdict

The core question is novel and well posed. The arXiv scan found no paper that gives a frozen model in-context evidence about its own behaviour (a record or self-run probes) and measures the effect on NSG. The control design (R, P, K, mismatched record, exact-match exclusion, probed/unprobed decomposition) is stronger than in most workshop papers.

The plan is about three times too large for a credible single write-up. Two of its components (H4 / Self-CTRL, and the self-specificity framing) are weaker than the rest. One recent paper (Sept 2026) threatens the metric itself and is not cited. Recommendation: run a focused version (Section 4) on one benchmark, and make the **probe-distance decomposition** the headline contribution.

## 2. Fact check of cited claims

| Claim in proposal | Status | Source checked |
|---|---|---|
| Self-CTRL raises auditor refusal-prediction accuracy 36% → 92% | ✓ (explanation training 0.36 → 0.92) | arXiv 2606.18327 |
| Mayne: self-explanations address up to 1/3 of predictor errors, 18 models | ✓ | arXiv 2602.02639 |
| Self beats cross-model by 1.7–4.3 points | ✓ Table 2; but the Table 9 variant gives +0.8 to +4.4. Cite both or say "about 1–4 pp" | 2602.02639 |
| Qwen3-32B NSG 35.6% vs 77.8% oracle ceiling | ✓ | 2602.02639 §4 |
| Blandfort & Pawar r = +0.04 → +0.24 with item information | ✓, **but the proposal omits the key detail**: asking the same question about "capable AI agents in general" gives +0.28, and other models' self-answers predict the target at least as well | arXiv 2609.09899 |
| CHIVE: trained counterfactual prediction generalises; interpretability tools give no uplift | ✓ | 2608.16747 |
| 2609.17119 counterfactual self-explanations; Guo et al. introspective coupling | ✓ both exist | arXiv |

## 3. Main issues, ordered by severity

1. **The metric can be bypassed (missing citation).** "Limitations of Automated Simulatability: LLM Simulators Can Bypass Explanations" (arXiv 2609.08585) reports that LLM simulators mainly rely on task priors, with explanations producing only small changes. Mayne's tabular tasks have meaningful labels (diabetes YES/NO, income), which is the setting where this shortcut operates. Required additions:
   - Report the prior-only baseline (no explanation) per dataset, plus a "task-rule" baseline: a predictor that guesses the ground-truth label ignoring the model.
   - Report NSG separately for items where the model's answer disagrees with the ground-truth label. That is where an explanation of *this model* has to carry the signal.
2. **The R baseline will probably win, so pre-register it.** Raw single-feature flip results are very informative for d=2 composition items. D ≤ R is a likely outcome. The post should present it as the "explanations compress evidence" test, not bury it. The most interesting single number is NSG(D) − NSG(R) on unprobed items.
3. **Dataset structure does not support the test-cell grid as released.** In `combined_dataset.parquet`, 5,191 of the 6,000 tabular pairs are at Hamming distance 2, 780 at d=1 and 29 at d=0. Each original item has only about 2.2–2.4 counterfactuals (5.2 for heart disease, 6.3 for breast cancer). Per-item cells such as "d=1, feature not probed" will be nearly empty. Fix: regenerate a denser Hamming ball per explained item using the repo's own generator (enumerate all d=1 and d=2 neighbours that occur in the data), or restrict the cell analysis to heart disease and breast cancer.
4. **H4 (Self-CTRL gap recovered) is not a like-for-like comparison.** It uses a different model (Llama-based), a different auditor and a different metric (auditor-generated counterfactuals judged by an LM), and data release is uncertain. A "fraction of training gap recovered" computed across protocols would draw justified criticism. Move it to future work, or run only "B and D on Self-CTRL's released prompts" if the data exist, without the ratio.
5. **Self-specificity is likely null and should be framed that way.** Given Blandfort & Pawar's generic-agent result, B ≈ C is the prior expectation. Keep C as a control, as the proposal does. Do not spend budget on k-sweeps of C.
6. **X (external investigator) is CHIVE's setup.** CHIVE already shows that an external agent running prompt edits writes predictive explanations. D vs X is therefore a test of whether *self*-experimentation adds anything over investigation. Expect D ≈ X, and frame the method as "investigate-then-explain, run by the model on itself" rather than claiming self-knowledge.
7. **Scope.** 6 conditions × 5 baselines × 3 targets × 3 predictors × k and n sweeps × 2 benchmarks is roughly 100+ cells. For a LessWrong or Alignment Forum post, one target model, two predictors, about 8 conditions and 400–600 explained items with paired bootstrap CIs is more credible than a sparse large grid.
8. **Statistics.** No published estimate exists for the effect of in-context evidence on NSG. The proposal's risk table cites the self-vs-cross-model uplift (+0.8 to +4.4 pp, Mayne Tables 2 and 9) as "the published effect", but that measures a different contrast. The relevant published anchors are: explanation vs no explanation, +3.8 to +10.8 pp absolute and 11.0–36.5% NSG (Mayne §4.1; Qwen3-8B 72.15% → 80.30%, NSG 29.2%); headroom to the oracle ceiling (Qwen3-32B 35.6% measured vs 77.8% ceiling); and training methods, which give large gains on their own metrics (Self-CTRL +56 pp; CST +35 pp cue-based but +2 pp generic). The effect size of this intervention is therefore unknown, not known to be small. Pair everything regardless: the same items and predictor across conditions, with a paired bootstrap over explained items (not pairs, because pairs within an item are correlated). Pre-register the primary contrast (D vs 0 on unprobed items) to avoid a garden of forking paths.
9. **Smaller points.** Add a leakage check where the explanation states the label for a specific profile (the proposal's LLM-judge audit is fine). Fix the answer-then-explain order (Mayne randomises it). Keep thinking off or token-capped for Qwen3 so that K is meaningful.

## 4. Recommended focused version (what I propose to implement)

- **Benchmark:** Mayne tabular datasets with denser regenerated neighbourhoods. Moral Machines only if budget allows.
- **Target:** one model with a published Mayne NSG (e.g. Qwen3-8B: published NSG 29.2%, or Gemma-3-12B / Qwen3-4B), first reproducing its baseline NSG.
- **Predictors:** two from Mayne's set (e.g. gpt-oss-20b and Qwen3-32B), reported separately.
- **Conditions:** 0, A, B(k=20, nearest neighbours), C, D(n=5), D+V, R, P, K. X if budget allows.
- **Headline analysis:** NSG by test cell (probed / unprobed / composition), D vs R vs P. Targeting efficiency of self-chosen probes vs ground-truth influential features.
- **Deliverables:** code repo, results tables with CIs, 4–5 figures, a draft post.

## 5. What makes it "innovative enough"

What a reviewer would see as new is not "probing helps", which CHIVE implies. It is (a) a training-free self-experimentation protocol evaluated with NSG, (b) the decomposition into lookup, interpolation and generalisation, which turns "you could just run the counterfactual" from a fatal objection into a measured quantity, and (c) whether models pick the right features to probe, compared against ground-truth feature influence. A clean negative result on (b) is publishable too.

## 6. Update after the pilot (2026-10-31)

A 100-item pilot was run for $4.81: target Gemma-3-27B-it (Qwen3-8B is no longer on OpenRouter), predictors gpt-oss-20b and Qwen3-32B. The full report is in pilot_report.md. In brief:

- The baseline reproduces Mayne (NSG 34.5% vs 34.92% published).
- Self-chosen probes (D) raise NSG by +16.4 points, but random probes do as well, and the raw probe table beats every explanation.
- The NSG gain is mostly a shift in the predictor's "no change" base rate. Gemma keeps its answer on 89% of test items, and no evidence condition improves balanced accuracy.
- The model does choose more informative probes than chance (13.0% vs 7.4% flip the answer).

Revised recommendation: make flip-balanced evaluation the primary metric, use flip-enriched test sets, and screen targets for sensitivity before any full run. The base-rate finding is itself the most publishable result so far.
