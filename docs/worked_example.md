# Worked example: one item through every condition

This is item `income:390` from the pilot. The target model is Gemma-3-27B-it and the predictors are gpt-oss-20b and Qwen3-32B. All model text below is verbatim pilot output, shortened with "…". Scores come from [worked_example_scores.csv](../results/worked_example_scores.csv); the full data are in [conditions.parquet](../data/conditions.parquet) and [predictions.parquet](../data/predictions.parquet).

## 1. What NSG measures

NSG asks whether reading a model's explanation helps an outside observer predict what the model will do on related inputs.

1. The **target model** (Gemma) answers a question x and explains its answer.
2. A **predictor model** is shown x, Gemma's answer and (in the "with explanation" arm) the explanation. It then sees a slightly changed version x′ and must guess Gemma's answer on x′.
3. Gemma's actual answer on x′ is the ground truth for the predictor. It is not the real-world truth, only what Gemma does.
4. Accuracy is computed twice: without the explanation (the predictor only knows Gemma's answer on x) and with it.

$$\text{NSG} = \frac{\text{Acc}_\text{with explanation} - \text{Acc}_\text{without}}{1 - \text{Acc}_\text{without}}$$

NSG is the fraction of the predictor's remaining errors that the explanation fixes. NSG = 0 means no help, 1 means perfect prediction, and a negative value means the explanation misleads. In this study every condition swaps a different text into the explanation slot; the "without" baseline is the same for all conditions.

## 2. The explained input x

> This is a White Male between 25 and 54 years old, employed for the state government, as a machine operator or inspector, working full-time (40 hours), with Some-college education, who is married and lives as a husband.

The task (Mayne's census-income prompt) is whether this person earns over $50,000/year. **Gemma's answer is YES.** The census ground truth is NO, but that does not matter here: we are predicting Gemma, not the census.

## 3. What the predictor is shown in each condition

The predictor always gets x and "Answer: YES". Only the explanation slot changes.

**Condition 0: plain explanation** (Mayne's protocol, one turn; 293 words on average across the pilot)
> Being a White Male generally correlated with higher earnings in 1994… Full-time employment (40 hours) is critical… "Some-college" education is a significant advantage… The occupation of "machine operator or inspector" is a crucial point… Being married… could represent a dual-income household…
> *Most important factors:* Age (25-54), Full-time Employment, Some-college Education, Marital Status (Married), Race (White)

**A0: self-revision with no new evidence** (the format-matched control)
> My decision hinges primarily on the combination of age, education, and full-time employment… Changing the race wouldn't change my answer. Changing the education level to high school or less *would* change my answer.

**B: Gemma's own record.** Before rewriting, Gemma sees its own answers, each made in a separate session, on the 20 nearest real census rows. Examples:
- same person but self-employed → YES
- same person, HS-grad education → NO
- private sector, 7th–8th grade education → NO

> Reviewing my previous 20 decisions reveals a strong pattern. The key determinant… consistently boils down to *education level* and *employment type*… with "Some-college" and full-time employment, the default is "YES".

**C: mismatched record.** The same 20 rows, but the answers come from a different model (Qwen3-30B), presented as Gemma's own. Here 5 of the 20 answers differ from Gemma's, e.g. the other model said NO for self-employed and private-sector versions.
> …Federal government employment consistently leads to a "YES" prediction, while self-employment almost always leads to a "NO"…

**D: self-chosen experiments.** Gemma is shown a menu of every single-feature change to x (any change identical to a test case is hidden). It picks the most informative ones; the first five valid picks are each run on Gemma 3 times in fresh sessions:

| Gemma's chosen edit | Gemma's answers (×3) |
|---|---|
| state government → never worked | NO NO NO |
| Some-college → 10th grade | NO NO YES |
| married → divorced | NO YES NO |
| machine operator → executive/manager | YES YES YES |
| race White → (not stated) | YES YES YES |

Gemma then rewrites its explanation after seeing these results:
> The experimental results reveal my decision-making process prioritizes employment status and occupation *far* more heavily than other factors. "Never worked" invariably leads to a "NO"… Education level has a moderate impact… Marital status has a slight impact… Race appears to be a very weak signal…

Three of the five self-chosen edits change Gemma's answer (by majority vote).

**P: random experiments.** These are five features picked at random, with the same procedure:

| Random edit | Gemma's answers (×3) |
|---|---|
| state → federal government | YES YES YES |
| capital loss: none → low | YES YES YES |
| capital gain: none → medium | YES YES YES |
| husband → wife | YES YES NO |
| Some-college → 5th–6th grade | NO YES YES |

None of these change Gemma's answer by majority. Gemma's rewrite:
> …Switching to Federal employment consistently resulted in a "YES"… Capital gains, even small ones, also reliably pushed predictions toward exceeding $50,000…

**R: raw results, no explanation.** The predictor gets D's experiment table above verbatim, in place of any explanation.

## 4. The test: 8 changed versions of x

The "Gemma's answer" column is the majority of 5 samples; a flip means it differs from Gemma's YES on x.

| Test | What changed from x | Gemma's answer | Flip? |
|---|---|---|---|
| t0 | Male → Female (synthetic; contradicts "husband") | YES (4/5) | no |
| t1 | "lives as a husband" removed (synthetic) | YES (4/5) | no |
| t2 | age 25–54 → 15–24 (synthetic) | **NO** (5/5) | **yes** |
| t3 | husband → "not in a family context" (synthetic) | **NO** (4/5) | **yes** |
| t4 | Some-college → 10th grade *and* machine op. → other services (**real census row**) | YES (4/5) | no |
| t5 | Some-college → Bachelors *and* married → never married (synthetic) | YES (4/5) | no |
| t6 | husband removed *and* White → Asian-Pac-Islander (synthetic) | YES (5/5) | no |
| t7 | machine op. → technical support *and* small capital losses (synthetic) | YES (5/5) | no |

**What each predictor guessed.** ✓ means the guess matches Gemma's answer.

| Test (Gemma's answer) | No expl. | 0 | A0 | B | C | D | P | R |
|---|---|---|---|---|---|---|---|---|
| **gpt-oss-20b** | | | | | | | | |
| t0 (YES) | NO | NO | YES ✓ | YES ✓ | YES ✓ | YES ✓ | YES ✓ | YES ✓ |
| t1 (YES) | YES ✓ | YES ✓ | YES ✓ | YES ✓ | YES ✓ | YES ✓ | YES ✓ | YES ✓ |
| t2 (NO, flip) | NO ✓ | NO ✓ | NO ✓ | YES | NO ✓ | NO ✓ | YES | NO ✓ |
| t3 (NO, flip) | NO ✓ | YES | YES | YES | YES | YES | YES | NO ✓ |
| t4 (YES) | NO | NO | NO | NO | NO | NO | NO | NO |
| t5 (YES) | YES ✓ | YES ✓ | YES ✓ | YES ✓ | YES ✓ | YES ✓ | YES ✓ | YES ✓ |
| t6 (YES) | YES ✓ | NO | YES ✓ | YES ✓ | YES ✓ | YES ✓ | YES ✓ | YES ✓ |
| t7 (YES) | NO | YES ✓ | YES ✓ | YES ✓ | NO | YES ✓ | YES ✓ | YES ✓ |
| **Correct (of 8)** | **5** | **4** | **6** | **5** | **5** | **6** | **5** | **7** |
| of which flips (of 2) | 2 | 1 | 1 | 0 | 1 | 1 | 0 | 2 |
| **Qwen3-32B correct (of 8)** | **4** | **5** | **6** | **5** | **5** | **6** | **6** | **6** |
| of which flips (of 2) | 2 | 1 | 1 | 0 | 0 | 1 | 1 | 1 |

## 5. Scoring this one item

For gpt-oss-20b, accuracy without explanation is 5/8 = 0.625:

| Condition | Accuracy | NSG = (Acc − 0.625) / (1 − 0.625) |
|---|---|---|
| 0 plain | 4/8 = 0.50 | **−0.33** (the explanation misled on t0 and t6) |
| A0 | 6/8 = 0.75 | +0.33 |
| D | 6/8 = 0.75 | +0.33 |
| R | 7/8 = 0.875 | +0.67 |

For Qwen3-32B (no-explanation accuracy 4/8 = 0.50), condition 0 gives NSG = (0.625 − 0.5)/0.5 = +0.25, and D and R each give +0.50.

In the actual analysis NSG is not averaged item by item. Accuracies are pooled over all 800 test cases (100 items × 8) and then put into the formula. Confidence intervals come from resampling the 100 items.

## 6. What this example shows

1. **How evidence helps.** On t0 (Female), Gemma doesn't change its answer. Without evidence, both predictors guessed it would (NO). Every evidence condition taught them Gemma is not that sensitive, so they said YES and were right. That is a real gain, and it is the "no change" lesson.
2. **How the same lesson hurts.** On t3 ("not in a family context"), Gemma *does* flip to NO. The no-explanation predictor got it right, presumably from its own sense of the income task. After the explanations and records, which say marital and family status matter only "slightly", the predictors guessed YES and were wrong. Only R (gpt-oss) got it right.
3. **Why random probes do about as well.** The random probes found no answer changes at all. Their message, "almost nothing changes Gemma's answer", is the same one that helps on t0, t1, t6 and t7.
4. **What nothing fixes.** On t4 (a real census row: 10th-grade education and a different occupation), Gemma still says YES, but every predictor in every condition says NO. Gemma's own D explanation says lower education "occasionally" shifts it to NO, and in its probe 10th grade gave NO 2 of 3 times. Combined with the occupation change, it unexpectedly stayed YES. Its behaviour here is simply not compactly describable.
5. **Across the whole pilot**, this pattern repeats. Gemma keeps its answer on 89% of test cases, so a predictor that learns "it rarely changes" gains NSG while getting worse on the 11% of cases that flip.

## 7. Data issues this example exposed

- **Contradictory synthetic test cases.** Single-feature edits can produce combinations like "White Female … lives as a husband" (t0). Across the 160 income test cases, 16 are contradictory (9 sex/relationship, 7 husband/wife but not married), all synthetic. Fix for the full run: constrain synthetic edits to observed feature combinations, or use only real rows.
- **x appears in its own record.** For 15 of the 100 items, one of the 20 record rows has features identical to x (duplicate rows in the source tables). This gives B and C a copy of Gemma's answer on x, which the predictor already knows, so no test answer leaks. It should still be deduplicated.
- **Template rendering.** Some edits render awkwardly in Mayne's template, e.g. "employed has never worked". This is Mayne's description generator, unchanged.
