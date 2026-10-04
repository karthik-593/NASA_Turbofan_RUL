### Stage X — FD004

| config | critical RMSE [95% CI] | n engines | n splits | subpop 0 | subpop 1 | wall (min) |
|---|---|---|---|---|---|---|
| `runs/FD004/xgboost/cap090_w030` | 6.11 [5.64, 6.62] | 249 | 15 | 5.19 [4.73, 5.62] (n=148) | 7.26 [6.48, 8.08] (n=101) | 1.7 |
| `runs/FD004/xgboost/cap090_w060` | 6.71 [5.74, 7.91] | 249 | 15 | 4.86 [4.38, 5.39] (n=148) | 8.74 [7.00, 10.63] (n=101) | 1.9 |
| `runs/FD004/xgboost/cap090_w090` | 7.42 [6.07, 8.98] | 249 | 15 | 4.97 [4.59, 5.38] (n=148) | 9.98 [7.30, 12.46] (n=101) | 1.8 |
| `runs/FD004/xgboost/cap090_w030_hi_consistent_regime_onehot` | 6.11 [5.64, 6.64] | 249 | 15 | 5.02 [4.57, 5.47] (n=148) | 7.43 [6.66, 8.29] (n=101) | 2.1 |
| `runs/FD004/xgboost/cap090_w030_hi_consistent_regime_onehot_baseline` | 6.61 [6.15, 7.09] | 249 | 15 | 5.94 [5.43, 6.42] (n=148) | 7.48 [6.68, 8.31] (n=101) | 2.1 |
| `runs/FD004/xgboost/cap090_w030_hi_consistent_regime_onehot_subpop_prob` | 6.41 [5.93, 6.88] | 249 | 15 | 5.59 [5.07, 6.09] (n=148) | 7.45 [6.66, 8.24] (n=101) | 2.5 |

Paired comparisons (Δ = challenger − incumbent; better = CI entirely < 0; metric = critical RMSE unless stated):

| step | challenger | incumbent | Δ [95% CI] | engines challenger better | Wilcoxon p (n splits) | verdict |
|---|---|---|---|---|---|---|
| A2-ext | xgboost, cap 90, window 60 | xgboost, cap 90, window 30 | 0.60 [-0.18, 1.62] | 57% of 249 | 0.15 (15) | not different |
| A2-ext | xgboost, cap 90, window 90 | xgboost, cap 90, window 30 | 1.31 [-0.02, 2.78] | 53% of 249 | 0.041 (15) | not different |
| A5a | xgboost, cap 90, window 30, hi_consistent, regime_onehot, baseline | xgboost, cap 90, window 30, hi_consistent, regime_onehot | 0.50 [0.13, 0.82] | 36% of 249 | 0.022 (15) | worse |
| A5b | xgboost, cap 90, window 30, hi_consistent, regime_onehot, subpop_prob | xgboost, cap 90, window 30, hi_consistent, regime_onehot | 0.30 [-0.02, 0.62] | 39% of 249 | 0.083 (15) | not different |

A4 — subpopulation from the first N cycles (logistic regression, AUC [95% CI], strong = CI lower bound ≥ 0.80; subpopulation sizes {'0': 148, '1': 101}):

| N cycles | AUC [95% CI] | per repeat | n engines | strong | shuffled-label AUC [95% CI] | control holds 0.5 |
|---|---|---|---|---|---|---|
| 30 | 0.995 [0.989, 0.999] | 0.994, 0.993, 0.995 | 249 | yes | 0.475 [0.405, 0.545] | yes |
| 50 | 0.997 [0.992, 1.000] | 0.996, 0.996, 0.997 | 249 | yes | 0.497 [0.419, 0.570] | yes |
| 100 | 1.000 [1.000, 1.000] | 1.000, 1.000, 1.000 | 249 | yes | 0.528 [0.455, 0.600] | yes |

Winners: A2-ext: xgboost, cap 90, window 30; A5a: xgboost, cap 90, window 30, hi_consistent, regime_onehot; A5b: xgboost, cap 90, window 30, hi_consistent, regime_onehot; stage_X: xgboost, cap 90, window 30, hi_consistent, regime_onehot
- Stage A adopted spec (re-run under current code where compared): xgboost, cap 90, window 30, hi_consistent, regime_onehot
- A4 at N = 30: AUC 0.995 [0.989, 0.999], shuffled-label control 0.475 [0.405, 0.545] -> gate passed
