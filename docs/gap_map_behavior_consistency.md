# Gap map: can a model make its own decisions more consistent by studying its own behaviour in context?

*Literature check, 2026-11-01. Every claim links to its source. The screening table (404 papers) is [lit_behavior_screened.csv](../results/lit_behavior_screened.csv).*

## 1. The idea, split into testable components

A frozen model **(A)** answers systematic variations of a decision input, or reviews its own past decisions. It then **(B)** writes explicit decision criteria from that evidence and **(C)** keeps those criteria in context when it answers new cases. The aim **(D)** is that its decision policy becomes more consistent: it follows its own stated rule, gives stable answers, and is invariant to features it says don't matter. This must not come from collapsing to a trivial answer.

This is the **behaviour side** of self-consistency. Self-CTRL names the two directions: explanation training makes generated rules predictive of responses, and behaviour training makes responses match the generated rule ([Pres et al. 2026](https://arxiv.org/abs/2606.18327)). The pilot tested the explanation side, with behaviour held fixed.

## 2. How the search was done

- 34 arXiv queries across rule induction, self-consistency, decision consistency, counterfactual invariance, precedent grounding, self-generated principles, stated vs revealed preferences, explanation consistency and LLM-judge rubrics. This gave 404 unique papers.
- Each abstract was scored by an LLM on components A–D, training-free status and a 0–3 relevance score. Then 70 shortlisted papers were read by hand, plus targeted web searches.
- Self-CTRL's full text was checked for any in-context baseline or future-work mention.
- **Limits:** arXiv plus web search only (no Semantic Scholar/OpenAlex, no ACL-only or non-arXiv venues); judgments based mostly on abstracts. "Not found" here means not found by this search, not proven absent.

## 3. What exists, by lineage

| Lineage | Key papers | A own behaviour | B explicit criteria | C applied in context | D consistency goal | Training-free | How it differs from the idea |
|---|---|---|---|---|---|---|---|
| **Behaviour-side consistency training** | [Self-CTRL](https://arxiv.org/abs/2606.18327) (behaviour direction); [Political Consistency Training](https://arxiv.org/abs/2605.22771); [EC-finetuning](https://arxiv.org/abs/2401.13986); [REPAIR / logical preference consistency](https://arxiv.org/abs/2410.02205) | ✓ | ✓ (Self-CTRL) | — (weights change) | ✓ | ✗ | Needs RL or fine-tuning. Self-CTRL's two baselines are also RL-trained one-sided rewards, not in-context methods. |
| **Training-free consistency over the model's own answers** | [BeliefBank](https://arxiv.org/abs/2109.14723); [ConCoRD](https://arxiv.org/abs/2211.11875); [REFLEX](https://arxiv.org/abs/2305.14250); [Chain of Guidance](https://arxiv.org/abs/2502.15924) | ✓ (answers on related questions or paraphrases) | ✗ | partly (BeliefBank's feedback component queries the model with known beliefs as context) | ✓ | ✓ (CoG also fine-tunes) | Factual QA beliefs, not decision criteria. Consistency is enforced by an external solver (MaxSAT, NLI) or answer selection, not by the model writing and applying its own rule. |
| **Rule induction, then deduction** | [Hypotheses-to-Theories](https://arxiv.org/abs/2310.07064); [Loo et al. 2025](https://arxiv.org/abs/2507.03876); [Robust Rule Induction / SRR](https://arxiv.org/abs/2502.16169) | ✗ (ground-truth labels) | ✓ | ✓ | accuracy (Loo et al. also report self-consistency) | ✓ | The evidence is labelled examples of an external concept, not the model's own behaviour. The goal is learning the true rule, not making the model's own policy consistent. |
| **Precedent grounding** | [Case Law Grounding](https://arxiv.org/abs/2310.07019); [CADA](https://arxiv.org/abs/2601.08000) | ✗ (community precedents) | ✗ | ✓ (CLG prompting) | ✓ | ✓ (CLG) / ✗ (CADA) | Past decisions come from humans or communities, not from the model's own probing. |
| **Say–do gap measurement** | [SNCA](https://arxiv.org/abs/2604.09189); [GCA, VLMs](https://arxiv.org/abs/2604.06422); [stated vs revealed preferences](https://arxiv.org/abs/2506.00751); [Doing What They Say](https://arxiv.org/abs/2606.00476) | ✓ | ✓ (rules elicited) | ✗ (SNCA deliberately keeps behaviour uninfluenced by the stated rule) | ✓ (measured) | ✓ | Measures the gap but does not intervene. The exception is Doing What They Say, which injects an *externally given* rule. |
| **Prompt-level stabilisation without self-evidence** | [CKM, Fact–Heuristic–Emotion](https://arxiv.org/abs/2607.24765); self-consistency decoding | ✗ | ✗ | structured reasoning state | ✓ | ✓ | Reduces run-to-run variability by structuring reasoning, with no self-derived criteria. |
| **Explanation-side, in context** | the pilot (this project) | ✓ | ✓ | ✗ (behaviour fixed) | explanation faithfulness | ✓ | The complementary direction. |

## 4. The gap

**No paper found combines A + B + C + D**: a frozen model probing its own decisions on systematic variations, writing its own criteria, and applying them in context to new cases, evaluated on decision-policy consistency with controls for degenerate consistency. The nearest neighbours each miss at least one component:

- **BeliefBank feedback** has A, C and D, but no explicit criteria, and the domain is factual beliefs.
- **Hypotheses-to-Theories** has B and C, but its evidence is ground-truth labels and its goal is accuracy.
- **Case Law Grounding** has C and D, but uses human precedents.
- **Self-CTRL behaviour training** has A, B and D, but needs RL.

## 5. Evidence from neighbours that predicts how it might go

| Finding | Source | Implication for the idea |
|---|---|---|
| Models violate their own stated rules often. GPT-5-mini violates its stated colour rule in nearly 60% of strong-prior cases; frontier models' self-stated safety policies diverge from their behaviour. | [GCA](https://arxiv.org/abs/2604.06422); [SNCA](https://arxiv.org/abs/2604.09189) | There is room to improve, so the baseline problem is real. |
| Injecting the decision rule into the prompt did not reliably raise adherence. The limiting factor was applying the rule, not knowing it. | [Doing What They Say](https://arxiv.org/abs/2606.00476) | A self-written rule may also fail to change behaviour. A rule supplied by another model (or a human) is a necessary control. |
| Behaviour updated toward vague rules drifts to a generic default. A consistency objective can be met by vague rules, and degenerate "consistent" fixed points must be blocked. | [Self-CTRL](https://arxiv.org/abs/2606.18327) §4, App. E | Score rule specificity, and guard against "always YES" consistency. This matches the pilot's base-rate lesson. |
| When behaviour becomes more predictable in general, NSG credits the no-explanation baseline; NSG was designed for fixed behaviour. | [Self-CTRL](https://arxiv.org/abs/2606.18327) App. J | NSG is the wrong primary metric for the behaviour side. Use rule adherence, stability and invariance, plus guardrails. |
| With rule elicitation in the prompt, 96.3% of classifications matched the model's self-reported rule. In another experiment, eliciting rules lowered classification accuracy (0.737 → 0.687). The evidence there was labelled examples, not the model's own behaviour. | [Loo et al. 2025](https://arxiv.org/abs/2507.03876) | Self-stated rules can be followed closely, but there may be a consistency vs accuracy trade-off; report both. |
| Models can reverse a decision when their own prior answer is fed back as context. | [CKM](https://arxiv.org/abs/2607.24765) | Putting self-evidence in context can destabilise as well as stabilise. Measure flips relative to baseline. |
| Grounding LLM decisions in past decisions (precedents) gave 20.8–32.9 pp higher accuracy against each community's own judgments, in 4 of 5 groups, compared with rule-based (constitutional) grounding. | [Case Law Grounding](https://arxiv.org/abs/2310.07019) | A record of concrete past cases (condition B) may beat abstract criteria. Compare "record only", "criteria only" and "both". |

## 6. What a study would need to show to fill the gap

1. **Primary outcome.** Rule adherence of new answers: an independent judge applies the model's own written criteria, and we check agreement. Secondary outcomes are stability across samples and orderings, and invariance to features the model calls irrelevant.
2. **Guardrails.** The answer distribution and degenerate-policy checks, how much behaviour changed from baseline, task accuracy against ground truth, and rule specificity.
3. **Controls:**
   - self-chosen vs random variations;
   - own record vs another model's record;
   - criteria written with no evidence (A0);
   - criteria written by a different model from the same evidence (does it need to be *self*-derived?);
   - record-only vs criteria-only vs both.
4. **Domains.** The Mayne tabular tasks, which come with ready-made variation generators. If available, a refusal setting comparable to Self-CTRL's behaviour-training result (refusal accuracy 0.36 → 0.62).

## 7. Caveats

- Searched arXiv and the web only. Work in ACL Anthology-only or other venues may be missed, and judgments come mostly from abstracts.
- The rule-induction and say–do literatures are large and fast-moving. A second pass with Semantic Scholar citation chaining from Self-CTRL, BeliefBank and Case Law Grounding would be the next check.
