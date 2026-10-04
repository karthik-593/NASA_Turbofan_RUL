### Stage X — FD001

| config | critical RMSE [95% CI] | n engines | n splits | subpop 0 | subpop 1 | wall (min) |
|---|---|---|---|---|---|---|
| `runs/FD001/xgboost/cap090_w045` | 3.97 [3.52, 4.50] | 100 | 15 | 4.11 [3.43, 4.89] (n=60) | 3.75 [3.31, 4.24] (n=40) | 2.4 |
| `runs/FD001/xgboost/cap090_w060` | 3.76 [3.39, 4.17] | 100 | 15 | 3.89 [3.34, 4.51] (n=60) | 3.56 [3.21, 3.91] (n=40) | 1.9 |
| `runs/FD001/xgboost/cap090_w090` | 4.13 [3.77, 4.48] | 100 | 15 | 4.05 [3.62, 4.51] (n=60) | 4.24 [3.74, 4.74] (n=40) | 2.0 |
| `runs/FD001/xgboost/cap090_w045_hi_pooled` | 3.41 [3.06, 3.73] | 100 | 15 | 3.40 [2.95, 3.84] (n=60) | 3.42 [2.96, 3.91] (n=40) | 2.0 |
| `runs/FD001/xgboost/cap090_w045_hi_pooled_baseline` | 3.40 [3.10, 3.70] | 100 | 15 | 3.56 [3.13, 3.95] (n=60) | 3.15 [2.77, 3.53] (n=40) | 1.7 |

Paired comparisons (Δ = challenger − incumbent; better = CI entirely < 0; metric = critical RMSE unless stated):

| step | challenger | incumbent | Δ [95% CI] | engines challenger better | Wilcoxon p (n splits) | verdict |
|---|---|---|---|---|---|---|
| A2-ext | xgboost, cap 90, window 60 | xgboost, cap 90, window 45 | -0.21 [-0.46, 0.05] | 56% of 100 | 0.11 (15) | not different |
| A2-ext | xgboost, cap 90, window 90 | xgboost, cap 90, window 45 | 0.16 [-0.27, 0.55] | 45% of 100 | 0.39 (15) | not different |
| A5a | xgboost, cap 90, window 45, hi_pooled, baseline | xgboost, cap 90, window 45, hi_pooled | -0.01 [-0.21, 0.22] | 50% of 100 | 0.76 (15) | not different |

A4 — subpopulation from the first N cycles (logistic regression, AUC [95% CI], strong = CI lower bound ≥ 0.80; subpopulation sizes {'0': 60, '1': 40}):

| N cycles | AUC [95% CI] | per repeat | n engines | strong | shuffled-label AUC [95% CI] | control holds 0.5 |
|---|---|---|---|---|---|---|
| 30 | 0.658 [0.543, 0.768] | 0.680, 0.661, 0.625 | 100 | no | 0.591 [0.484, 0.700] | yes |
| 50 | 0.723 [0.620, 0.813] | 0.758, 0.682, 0.693 | 100 | no | 0.527 [0.417, 0.646] | yes |
| 100 | 0.951 [0.909, 0.985] | 0.940, 0.948, 0.957 | 100 | yes | 0.550 [0.427, 0.660] | yes |

Winners: A2-ext: xgboost, cap 90, window 45; A5a: xgboost, cap 90, window 45, hi_pooled; stage_X: xgboost, cap 90, window 45, hi_pooled
- Stage A adopted spec (re-run under current code where compared): xgboost, cap 90, window 45, hi_pooled
