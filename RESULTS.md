# C+T1 follow-up results

Reviewed: 26 October 2026. **Real training and prediction review passed.**

C+T1 (`CT1`) is a newly trained Random Forest using C's four extra calculations with T1's settings. It is the preferred candidate in this exploratory follow-up because it has the highest mean grouped CV Macro F1 among B, C, T1 and CT1. The historical frozen T1 decision and test result remain unchanged.

## What changed

The original fifteen measurements stay unchanged. Four inputs are added: forward/backward packet ratio, forward/backward byte ratio, forward packet share, and an indicator for no backward packets. The recorded C formulas are reused, including zero when a ratio denominator is zero. T1's 80 trees, depth 20, minimum leaf size 2, balanced fitting-side weights, seed 42, and two CPU workers are unchanged. There is no merging of old model files.

The same split and three grouped folds were used. Heartbleed's twenty rows remain fitting-only; UNLABELED rows are excluded. Three CV models and one original-training-only model completed. The consumed test was neither loaded nor scored. One earlier fit attempt failed during prediction saving because storage filled; it published no completion and was retried unchanged.

## Same-fold comparison

| Model | Inputs | Mean CV Macro F1 | Sample SD (percentage points) | Mean benign false alarms |
|---|---:|---:|---:|---:|
| CT1 | 19 | 72.30% | 7.57 | 1.80% |
| T1 | 15 | 71.43% | 8.49 | 2.20% |
| C | 19 | 66.10% | 8.67 | 2.56% |
| B | 15 | 65.37% | 8.01 | 2.62% |

CT1 improves mean CV Macro F1 over T1 by **0.87 percentage points**. Its sample variation and mean false-alarm rate are lower. It improved two folds and slightly regressed on one. Three folds do not establish a reliable general improvement; the second fold remains difficult. Full-precision scores decided the ranking; there was no tie. Historical timing values came from a different runtime and did not decide selection.

## Descriptive validation

| Model | Macro F1 | Accuracy | BENIGN false alarms |
|---|---:|---:|---:|
| T1 | 77.75% | 99.49% | 2,255 / 400,247 |
| CT1 | 78.03% | 99.50% | 2,254 / 400,247 |

Both models were fitted on 1,472,712 original training rows and this comparison uses the same 481,841 validation rows. Validation already informed development. CT1's validation gain is **0.29 percentage points**; it is not a fresh final-test result.

## Remaining weaknesses

Validation F1 is 67.70% for Bot (443 rows), 11.11% for Infiltration (11), 41.03% for XSS (19), and **0%** for SQL Injection (2). These rare-class supports are small and do not support strong detection claims. Heartbleed has zero held-out support, so its precision, recall and F1 are **N/A**, not zero. High 99.50% overall validation accuracy does not resolve these class weaknesses.

## Review and decision

An independent reconstruction matched all 1,954,533 candidate prediction rows to their saved source position, label, conversation, capture, group and partition. All 900 confusion-matrix cells, per-class metrics, accuracy, kappa, Macro F1, class weights, source counts, model settings and ordered inputs passed. All four completion records and their artifact hashes passed. The original test file is absent from the isolated execution inputs.

**Decision:** use CT1 as the preferred exploratory follow-up candidate and offer both models in the demonstration. Preserve historical T1 with its 85.77% independent final-test Macro F1. CT1's independent test status is **not evaluated**. Do not transfer T1's test number to CT1 or compare this CV score directly with the separate official-flow study.

The demo includes 31 selected real validation rows covering the fourteen available held-out classes, including correct predictions and mistakes. Both saved models reproduce all recorded example predictions. No IP addresses, conversation identifiers, or test examples are published. These selected examples illustrate behavior; they are not a benchmark.

## Evidence and provenance

[Independent prediction review](REVIEW.json), [ordered inputs and model hashes](demo/bundle/manifest.json), [model card](demo/bundle/MODEL_CARD.md), and [sample provenance](demo/bundle/samples/SAMPLE_DATA.md) are included here. Aggregate fold and class results are available in the linked project reviewer. Full research artifacts remain in the original project.
