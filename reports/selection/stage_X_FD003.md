### Stage X — FD003

| config | critical RMSE [95% CI] | n engines | n splits | subpop 0 | subpop 1 | wall (min) |
|---|---|---|---|---|---|---|
| `runs/FD003/xgboost/cap090_w045` | 3.81 [3.28, 4.44] | 100 | 15 | 4.42 [3.65, 5.24] (n=56) | 2.84 [2.51, 3.18] (n=44) | 1.9 |
| `runs/FD003/xgboost/cap090_w060` | 3.65 [3.25, 4.09] | 100 | 15 | 4.14 [3.55, 4.76] (n=56) | 2.92 [2.62, 3.21] (n=44) | 1.5 |
| `runs/FD003/xgboost/cap090_w090` | 4.10 [3.73, 4.47] | 100 | 15 | 4.50 [4.03, 5.00] (n=56) | 3.51 [3.07, 4.00] (n=44) | 1.5 |
| `runs/FD003/xgboost/cap090_w045_hi_consistent` | 3.45 [3.01, 3.98] | 100 | 15 | 3.99 [3.32, 4.68] (n=56) | 2.60 [2.31, 2.90] (n=44) | 1.8 |
| `runs/FD003/xgboost/cap090_w045_hi_consistent_baseline` | 4.08 [3.52, 4.68] | 100 | 15 | 4.35 [3.77, 4.87] (n=56) | 3.70 [2.75, 4.82] (n=44) | 1.5 |
| `runs/FD003/xgboost/cap090_w045_hi_consistent_subpop_prob` | 3.66 [3.22, 4.25] | 100 | 15 | 4.16 [3.48, 4.94] (n=56) | 2.90 [2.60, 3.20] (n=44) | 1.7 |

Paired comparisons (Δ = challenger − incumbent; better = CI entirely < 0; metric = critical RMSE unless stated):

| step | challenger | incumbent | Δ [95% CI] | engines challenger better | Wilcoxon p (n splits) | verdict |
|---|---|---|---|---|---|---|
| A2-ext | xgboost, cap 90, window 60 | xgboost, cap 90, window 45 | -0.15 [-0.67, 0.24] | 49% of 100 | 0.45 (15) | not different |
| A2-ext | xgboost, cap 90, window 90 | xgboost, cap 90, window 45 | 0.29 [-0.19, 0.72] | 35% of 100 | 0.035 (15) | not different |
| A5a | xgboost, cap 90, window 45, hi_consistent, baseline | xgboost, cap 90, window 45, hi_consistent | 0.63 [0.17, 1.19] | 30% of 100 | 0.0054 (15) | worse |
| A5b | xgboost, cap 90, window 45, hi_consistent, subpop_prob | xgboost, cap 90, window 45, hi_consistent | 0.21 [0.10, 0.34] | 29% of 100 | 0.022 (15) | worse |

A4 — subpopulation from the first N cycles (logistic regression, AUC [95% CI], strong = CI lower bound ≥ 0.80; subpopulation sizes {'0': 56, '1': 44}):

| N cycles | AUC [95% CI] | per repeat | n engines | strong | shuffled-label AUC [95% CI] | control holds 0.5 |
|---|---|---|---|---|---|---|
| 30 | 0.997 [0.989, 1.000] | 0.992, 0.996, 0.995 | 100 | yes | 0.463 [0.348, 0.578] | yes |
| 50 | 0.994 [0.983, 1.000] | 0.992, 0.996, 0.996 | 100 | yes | 0.349 [0.247, 0.457] | NO |
| 100 | 1.000 [1.000, 1.000] | 1.000, 1.000, 1.000 | 100 | yes | 0.448 [0.337, 0.566] | yes |

Winners: A2-ext: xgboost, cap 90, window 45; A5a: xgboost, cap 90, window 45, hi_consistent; A5b: xgboost, cap 90, window 45, hi_consistent; stage_X: xgboost, cap 90, window 45, hi_consistent
- Stage A adopted spec (re-run under current code where compared): xgboost, cap 90, window 45, hi_consistent
- A4 at N = 30: AUC 0.997 [0.989, 1.000], shuffled-label control 0.463 [0.348, 0.578] -> gate passed
