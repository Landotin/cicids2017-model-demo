# CT1 model card

CT1 is an exploratory Random Forest fit that uses the original fifteen saved measurements plus four recorded C formulas. It uses 80 trees, depth 20, minimum leaf size 2, balanced fitting-side weights, seed 42 and two CPU workers. The model was fitted on the original training partition only. Its three saved grouped folds and validation score are descriptive; validation informed development. CT1 has not been scored on an independent test set. Heartbleed remains in fitting, with held-out scores shown as N/A.

T1 is included only as the preserved frozen 15-input model when its recorded model file is accessible and hash-verified. Its historical final-test metrics describe T1 only and are not CT1 evidence.
