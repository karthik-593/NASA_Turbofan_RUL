### Stage A — FD001

| config | critical RMSE [95% CI] | n engines | subpop 0 | subpop 1 | wall (min) |
|---|---|---|---|---|---|
| `A/FD001/cap090` | 4.98 [4.53, 5.45] | 100 | 4.63 [4.12, 5.12] (n=60) | 5.46 [4.70, 6.26] (n=40) | 8.8 |
| `A/FD001/cap105` | 4.99 [4.58, 5.44] | 100 | 4.69 [4.24, 5.15] (n=60) | 5.41 [4.67, 6.21] (n=40) | 8.3 |
| `A/FD001/cap125` | 5.19 [4.78, 5.64] | 100 | 4.76 [4.31, 5.20] (n=60) | 5.78 [5.00, 6.60] (n=40) | 8.5 |
| `A/FD001/cap140` | 5.31 [4.89, 5.79] | 100 | 4.90 [4.47, 5.33] (n=60) | 5.88 [5.00, 6.77] (n=40) | 8.2 |
| `A/FD001/cap090_w30` | 4.02 [3.72, 4.30] | 100 | 3.93 [3.58, 4.26] (n=60) | 4.13 [3.66, 4.61] (n=40) | 8.4 |
| `A/FD001/cap090_w45` | 3.98 [3.51, 4.53] | 100 | 4.12 [3.41, 4.89] (n=60) | 3.77 [3.29, 4.28] (n=40) | 6.1 |
| `A/FD001/cap090_w45_hi_consistent` | 3.51 [3.18, 3.84] | 100 | 3.58 [3.11, 4.05] (n=60) | 3.41 [3.00, 3.82] (n=40) | 9.0 |
| `A/FD001/cap090_w45_hi_pooled` | 3.41 [3.05, 3.76] | 100 | 3.39 [2.92, 3.85] (n=60) | 3.44 [2.97, 3.95] (n=40) | 9.1 |
| `A/FD001/cap090_w45_sensors_plus` | 4.02 [3.56, 4.56] | 100 | 4.17 [3.49, 4.95] (n=60) | 3.79 [3.34, 4.27] (n=40) | 9.0 |
| `A/FD001/cap090_w45_combined` | 3.41 [3.05, 3.76] | 100 | 3.39 [2.92, 3.85] (n=60) | 3.44 [2.97, 3.95] (n=40) | 0.6 |

Paired comparisons (Δ = challenger − incumbent, critical RMSE; better = CI entirely < 0):

| step | challenger | incumbent | Δ [95% CI] | engines challenger better | Wilcoxon p (n splits) | verdict |
|---|---|---|---|---|---|---|
| A1 | cap 90 | cap 125 | -0.22 [-0.38, -0.07] | 65% of 100 | 0.012 (15) | **better** |
| A1 | cap 105 | cap 125 | -0.20 [-0.33, -0.09] | 64% of 100 | 0.041 (15) | **better** |
| A1 | cap 140 | cap 125 | 0.12 [0.01, 0.26] | 43% of 100 | 0.14 (15) | worse |
| A2 | cap 90, window 30 | cap 90, window 20 | -0.96 [-1.38, -0.56] | 72% of 100 | 0.00031 (15) | **better** |
| A2 | cap 90, window 45 | cap 90, window 20 | -1.00 [-1.60, -0.36] | 71% of 100 | 0.0026 (15) | **better** |
| A3 | cap 90, window 45 + hi_consistent | cap 90, window 45 (base features) | -0.47 [-0.84, -0.18] | 67% of 100 | 6.1e-05 (15) | **better** |
| A3 | cap 90, window 45 + hi_pooled | cap 90, window 45 (base features) | -0.57 [-0.91, -0.24] | 61% of 100 | 6.1e-05 (15) | **better** |
| A3 | cap 90, window 45 + sensors_plus | cap 90, window 45 (base features) | 0.04 [-0.02, 0.10] | 47% of 100 | 0.17 (15) | not different |
| A3-combined | cap 90, window 45 + hi_pooled | cap 90, window 45 (base features) | -0.57 [-0.95, -0.26] | 61% of 100 | 6.1e-05 (15) | **better** |

Decision check for each adopted change (caught % with lead ≥ 20 at matched mean wasted life; Δ = challenger − incumbent, positive = challenger catches more):

| step | challenger vs incumbent | budget (cycles) | caught % challenger | caught % incumbent | Δ [95% CI] | n engines |
|---|---|---|---|---|---|---|
| A1 | cap 90 vs cap 125 | 20 | 47.1 | 47.0 | 0.1 [-2.6, 2.9] | 100 |
| A1 | cap 90 vs cap 125 | 30 | 95.8 | 95.5 | 0.3 [-1.4, 2.1] | 100 |
| A1 | cap 90 vs cap 125 | 40 | 100.0 | 100.0 | 0.0 [0.0, 0.0] | 100 |
| A2 | cap 90, window 45 vs cap 90, window 20 | 20 | 49.8 | 47.1 | 2.7 [-2.8, 9.0] | 100 |
| A2 | cap 90, window 45 vs cap 90, window 20 | 30 | 97.2 | 95.8 | 1.5 [-1.5, 5.4] | 100 |
| A2 | cap 90, window 45 vs cap 90, window 20 | 40 | 100.0 | 100.0 | 0.0 [0.0, 0.0] | 100 |
| A3 | cap 90, window 45 + hi_pooled vs cap 90, window 45 (base features) | 20 | 51.4 | 49.8 | 1.6 [-3.2, 6.3] | 100 |
| A3 | cap 90, window 45 + hi_pooled vs cap 90, window 45 (base features) | 30 | 96.7 | 97.2 | -0.5 [-2.5, 1.7] | 100 |
| A3 | cap 90, window 45 + hi_pooled vs cap 90, window 45 (base features) | 40 | 100.0 | 100.0 | 0.0 [0.0, 0.0] | 100 |

A4 — subpopulation from the first N cycles (logistic regression, AUC [95% CI], strong = CI lower bound ≥ 0.80; subpopulation sizes {'0': 60, '1': 40}):

| N cycles | AUC [95% CI] | per repeat | n engines | strong |
|---|---|---|---|---|
| 30 | 0.658 [0.543, 0.768] | 0.680, 0.661, 0.625 | 100 | no |
| 50 | 0.723 [0.620, 0.813] | 0.758, 0.682, 0.693 | 100 | no |
| 100 | 0.951 [0.909, 0.985] | 0.940, 0.948, 0.957 | 100 | yes |

Winners: A1: cap 90; A2: cap 90, window 45; A3: cap 90, window 45 + hi_pooled; stage_A: cap 90, window 45 + hi_pooled
- A3-combined: both health indices adopted; hi_pooled (lower point estimate) is the one combined, as the two share the same three columns
