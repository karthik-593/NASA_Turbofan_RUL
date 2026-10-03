### Stage A — FD003

| config | critical RMSE [95% CI] | n engines | subpop 0 | subpop 1 | wall (min) |
|---|---|---|---|---|---|
| `A/FD003/cap090` | 4.68 [4.12, 5.25] | 100 | 5.59 [4.86, 6.38] (n=56) | 3.17 [2.90, 3.46] (n=44) | 9.4 |
| `A/FD003/cap105` | 4.78 [4.19, 5.39] | 100 | 5.70 [4.90, 6.61] (n=56) | 3.27 [3.00, 3.55] (n=44) | 9.6 |
| `A/FD003/cap125` | 4.99 [4.40, 5.61] | 100 | 5.90 [5.12, 6.78] (n=56) | 3.50 [3.20, 3.79] (n=44) | 9.2 |
| `A/FD003/cap140` | 5.13 [4.54, 5.72] | 100 | 6.02 [5.24, 6.89] (n=56) | 3.69 [3.36, 3.99] (n=44) | 9.6 |
| `A/FD003/cap090_w30` | 4.03 [3.66, 4.42] | 100 | 4.66 [4.18, 5.17] (n=56) | 3.04 [2.71, 3.38] (n=44) | 8.2 |
| `A/FD003/cap090_w45` | 3.80 [3.27, 4.41] | 100 | 4.41 [3.62, 5.23] (n=56) | 2.85 [2.53, 3.19] (n=44) | 9.8 |
| `A/FD003/cap090_w45_hi_consistent` | 3.55 [3.12, 4.08] | 100 | 4.12 [3.45, 4.80] (n=56) | 2.66 [2.35, 2.97] (n=44) | 10.4 |
| `A/FD003/cap090_w45_hi_pooled` | 3.71 [3.24, 4.29] | 100 | 4.29 [3.62, 5.07] (n=56) | 2.80 [2.47, 3.16] (n=44) | 10.5 |
| `A/FD003/cap090_w45_sensors_plus` | 3.85 [3.33, 4.48] | 100 | 4.49 [3.70, 5.33] (n=56) | 2.84 [2.56, 3.13] (n=44) | 10.8 |

Paired comparisons (Δ = challenger − incumbent, critical RMSE; better = CI entirely < 0):

| step | challenger | incumbent | Δ [95% CI] | engines challenger better | Wilcoxon p (n splits) | verdict |
|---|---|---|---|---|---|---|
| A1 | cap 90 | cap 125 | -0.31 [-0.50, -0.13] | 69% of 100 | 0.0043 (15) | **better** |
| A1 | cap 105 | cap 125 | -0.21 [-0.36, -0.08] | 67% of 100 | 0.0015 (15) | **better** |
| A1 | cap 140 | cap 125 | 0.13 [0.05, 0.21] | 27% of 100 | 0.041 (15) | worse |
| A2 | cap 90, window 30 | cap 90, window 20 | -0.65 [-1.16, -0.20] | 58% of 100 | 0.00018 (15) | **better** |
| A2 | cap 90, window 45 | cap 90, window 20 | -0.88 [-1.58, -0.17] | 74% of 100 | 0.00085 (15) | **better** |
| A3 | cap 90, window 45 + hi_consistent | cap 90, window 45 (base features) | -0.25 [-0.43, -0.06] | 61% of 100 | 0.0054 (15) | **better** |
| A3 | cap 90, window 45 + hi_pooled | cap 90, window 45 (base features) | -0.09 [-0.25, 0.06] | 47% of 100 | 0.095 (15) | not different |
| A3 | cap 90, window 45 + sensors_plus | cap 90, window 45 (base features) | 0.05 [-0.04, 0.13] | 37% of 100 | 0.39 (15) | not different |

Decision check for each adopted change (caught % with lead ≥ 20 at matched mean wasted life; Δ = challenger − incumbent, positive = challenger catches more):

| step | challenger vs incumbent | budget (cycles) | caught % challenger | caught % incumbent | Δ [95% CI] | n engines |
|---|---|---|---|---|---|---|
| A1 | cap 90 vs cap 125 | 20 | 47.7 | 47.2 | 0.5 [-1.6, 2.8] | 100 |
| A1 | cap 90 vs cap 125 | 30 | 94.7 | 94.7 | -0.0 [-1.4, 1.5] | 100 |
| A1 | cap 90 vs cap 125 | 40 | 100.0 | 99.0 | 0.9 [-0.1, 2.3] | 100 |
| A2 | cap 90, window 45 vs cap 90, window 20 | 20 | 51.5 | 47.7 | 3.8 [-1.4, 8.9] | 100 |
| A2 | cap 90, window 45 vs cap 90, window 20 | 30 | 98.7 | 94.7 | 4.0 [0.9, 7.1] | 100 |
| A2 | cap 90, window 45 vs cap 90, window 20 | 40 | 100.0 | 100.0 | 0.0 [0.0, 0.7] | 100 |
| A3 | cap 90, window 45 + hi_consistent vs cap 90, window 45 (base features) | 20 | 51.7 | 51.5 | 0.2 [-4.6, 5.1] | 100 |
| A3 | cap 90, window 45 + hi_consistent vs cap 90, window 45 (base features) | 30 | 99.1 | 98.7 | 0.3 [-0.6, 1.4] | 100 |
| A3 | cap 90, window 45 + hi_consistent vs cap 90, window 45 (base features) | 40 | 100.0 | 100.0 | 0.0 [0.0, 0.0] | 100 |

A4 — subpopulation from the first N cycles (logistic regression, AUC [95% CI], strong = CI lower bound ≥ 0.80; subpopulation sizes {'0': 56, '1': 44}):

| N cycles | AUC [95% CI] | per repeat | n engines | strong |
|---|---|---|---|---|
| 30 | 0.997 [0.989, 1.000] | 0.992, 0.996, 0.995 | 100 | yes |
| 50 | 0.994 [0.983, 1.000] | 0.992, 0.996, 0.996 | 100 | yes |
| 100 | 1.000 [1.000, 1.000] | 1.000, 1.000, 1.000 | 100 | yes |

Winners: A1: cap 90; A2: cap 90, window 45; A3: cap 90, window 45 + hi_consistent; stage_A: cap 90, window 45 + hi_consistent
