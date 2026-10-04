### Stage X — FD002

| config | critical RMSE [95% CI] | n engines | n splits | subpop 0 | subpop 1 | wall (min) |
|---|---|---|---|---|---|---|
| `runs/FD002/xgboost/cap090_w045` | 4.14 [3.90, 4.40] | 260 | 15 | 4.08 [3.80, 4.39] (n=183) | 4.27 [3.84, 4.73] (n=77) | 2.1 |
| `runs/FD002/xgboost/cap090_w060` | 4.06 [3.79, 4.35] | 260 | 15 | 4.09 [3.74, 4.47] (n=183) | 3.99 [3.61, 4.39] (n=77) | 2.1 |
| `runs/FD002/xgboost/cap090_w090` | 4.34 [4.06, 4.65] | 260 | 15 | 4.36 [4.01, 4.76] (n=183) | 4.29 [3.87, 4.70] (n=77) | 2.2 |
| `runs/FD002/xgboost/cap090_w045_hi_consistent` | 3.94 [3.65, 4.24] | 260 | 15 | 3.79 [3.51, 4.11] (n=183) | 4.27 [3.68, 4.86] (n=77) | 2.5 |
| `runs/FD002/xgboost/cap090_w045_hi_consistent_baseline` | 4.30 [3.97, 4.66] | 260 | 15 | 4.33 [3.92, 4.80] (n=183) | 4.23 [3.79, 4.68] (n=77) | 2.3 |

Paired comparisons (Δ = challenger − incumbent; better = CI entirely < 0; metric = critical RMSE unless stated):

| step | challenger | incumbent | Δ [95% CI] | engines challenger better | Wilcoxon p (n splits) | verdict |
|---|---|---|---|---|---|---|
| A2-ext | xgboost, cap 90, window 60 | xgboost, cap 90, window 45 | -0.08 [-0.28, 0.13] | 56% of 260 | 0.36 (15) | not different |
| A2-ext | xgboost, cap 90, window 90 | xgboost, cap 90, window 45 | 0.20 [-0.06, 0.47] | 48% of 260 | 0.073 (15) | not different |
| A5a | xgboost, cap 90, window 45, hi_consistent, baseline | xgboost, cap 90, window 45, hi_consistent | 0.36 [0.11, 0.62] | 39% of 260 | 0.0012 (15) | worse |

A4 — subpopulation from the first N cycles (logistic regression, AUC [95% CI], strong = CI lower bound ≥ 0.80; subpopulation sizes {'0': 183, '1': 77}):

| N cycles | AUC [95% CI] | per repeat | n engines | strong | shuffled-label AUC [95% CI] | control holds 0.5 |
|---|---|---|---|---|---|---|
| 30 | 0.523 [0.445, 0.597] | 0.491, 0.507, 0.559 | 260 | no | 0.581 [0.507, 0.660] | NO |
| 50 | 0.589 [0.509, 0.663] | 0.578, 0.588, 0.597 | 260 | no | 0.541 [0.461, 0.626] | yes |
| 100 | 0.951 [0.922, 0.973] | 0.946, 0.945, 0.953 | 260 | yes | 0.555 [0.478, 0.627] | yes |

Winners: A2-ext: xgboost, cap 90, window 45; A5a: xgboost, cap 90, window 45, hi_consistent; stage_X: xgboost, cap 90, window 45, hi_consistent
- Stage A adopted spec (re-run under current code where compared): xgboost, cap 90, window 45, hi_consistent
