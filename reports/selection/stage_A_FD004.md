### Stage A — FD004

| config | critical RMSE [95% CI] | n engines | subpop 0 | subpop 1 | wall (min) |
|---|---|---|---|---|---|
| `A/FD004/cap090` | 6.58 [6.12, 7.05] | 249 | 6.15 [5.58, 6.72] (n=148) | 7.16 [6.54, 7.86] (n=101) | 20.9 |
| `A/FD004/cap105` | 6.78 [6.33, 7.25] | 249 | 6.22 [5.62, 6.76] (n=148) | 7.53 [6.87, 8.21] (n=101) | 21.0 |
| `A/FD004/cap125` | 7.12 [6.62, 7.64] | 249 | 6.36 [5.77, 6.92] (n=148) | 8.10 [7.29, 8.94] (n=101) | 17.9 |
| `A/FD004/cap140` | 7.32 [6.79, 7.86] | 249 | 6.32 [5.74, 6.88] (n=148) | 8.58 [7.71, 9.52] (n=101) | 21.3 |
| `A/FD004/cap090_w30` | 6.18 [5.67, 6.74] | 249 | 5.18 [4.73, 5.62] (n=148) | 7.40 [6.60, 8.28] (n=101) | 21.4 |
| `A/FD004/cap090_w45` | 6.37 [5.74, 7.02] | 249 | 4.94 [4.44, 5.47] (n=148) | 8.02 [6.95, 9.08] (n=101) | 6.0 |
| `A/FD004/cap090_w30_hi_consistent` | 5.99 [5.54, 6.48] | 249 | 4.97 [4.52, 5.43] (n=148) | 7.22 [6.51, 7.97] (n=101) | 22.5 |
| `A/FD004/cap090_w30_hi_pooled` | 6.25 [5.73, 6.80] | 249 | 5.15 [4.69, 5.59] (n=148) | 7.57 [6.68, 8.49] (n=101) | 23.0 |
| `A/FD004/cap090_w30_regime_onehot` | 6.10 [5.63, 6.60] | 249 | 5.14 [4.68, 5.57] (n=148) | 7.28 [6.49, 8.11] (n=101) | 21.8 |
| `A/FD004/cap090_w30_sensors_plus_fittable` | 6.66 [5.98, 7.48] | 249 | 5.21 [4.74, 5.64] (n=148) | 8.35 [7.15, 9.75] (n=101) | 24.3 |
| `A/FD004/cap090_w30_combined` | 5.98 [5.51, 6.47] | 249 | 4.96 [4.49, 5.43] (n=148) | 7.21 [6.47, 8.01] (n=101) | 22.8 |

Paired comparisons (Δ = challenger − incumbent, critical RMSE; better = CI entirely < 0):

| step | challenger | incumbent | Δ [95% CI] | engines challenger better | Wilcoxon p (n splits) | verdict |
|---|---|---|---|---|---|---|
| A1 | cap 90 | cap 125 | -0.54 [-0.70, -0.38] | 70% of 249 | 0.00043 (15) | **better** |
| A1 | cap 105 | cap 125 | -0.33 [-0.46, -0.22] | 63% of 249 | 0.00031 (15) | **better** |
| A1 | cap 140 | cap 125 | 0.21 [0.09, 0.33] | 40% of 249 | 0.022 (15) | worse |
| A2 | cap 90, window 30 | cap 90, window 20 | -0.40 [-0.73, -0.06] | 61% of 249 | 0.002 (15) | **better** |
| A2 | cap 90, window 45 | cap 90, window 20 | -0.21 [-0.77, 0.36] | 59% of 249 | 0.19 (15) | not different |
| A3 | cap 90, window 30 + hi_consistent | cap 90, window 30 (base features) | -0.19 [-0.34, -0.04] | 65% of 249 | 0.018 (15) | **better** |
| A3 | cap 90, window 30 + hi_pooled | cap 90, window 30 (base features) | 0.07 [-0.01, 0.15] | 55% of 249 | 0.33 (15) | not different |
| A3 | cap 90, window 30 + regime_onehot | cap 90, window 30 (base features) | -0.08 [-0.17, -0.01] | 60% of 249 | 0.12 (15) | **better** |
| A3-informational | cap 90, window 30 + sensors_plus_fittable (deviation) | cap 90, window 30 (base features) | 0.49 [0.03, 1.01] | 51% of 249 | 0.03 (15) | worse (informational, not eligible) |
| A3-combined | cap 90, window 30 + hi_consistent+regime_onehot | cap 90, window 30 (base features) | -0.20 [-0.36, -0.06] | 63% of 249 | 0.026 (15) | **better** |

Decision check for each adopted change (caught % with lead ≥ 20 at matched mean wasted life; Δ = challenger − incumbent, positive = challenger catches more):

| step | challenger vs incumbent | budget (cycles) | caught % challenger | caught % incumbent | Δ [95% CI] | n engines |
|---|---|---|---|---|---|---|
| A1 | cap 90 vs cap 125 | 20 | 36.2 | 36.6 | -0.4 [-2.1, 1.5] | 249 |
| A1 | cap 90 vs cap 125 | 30 | 86.7 | 85.0 | 1.7 [0.2, 3.1] | 249 |
| A1 | cap 90 vs cap 125 | 40 | 99.1 | 99.0 | 0.1 [-0.5, 0.6] | 249 |
| A2 | cap 90, window 30 vs cap 90, window 20 | 20 | 38.3 | 36.2 | 2.2 [-1.3, 5.7] | 249 |
| A2 | cap 90, window 30 vs cap 90, window 20 | 30 | 89.7 | 86.7 | 3.0 [0.1, 6.0] | 249 |
| A2 | cap 90, window 30 vs cap 90, window 20 | 40 | 99.2 | 99.1 | 0.2 [-0.8, 1.3] | 249 |
| A3 | cap 90, window 30 + hi_consistent+regime_onehot vs cap 90, window 30 (base features) | 20 | 37.0 | 38.3 | -1.4 [-3.4, 0.6] | 249 |
| A3 | cap 90, window 30 + hi_consistent+regime_onehot vs cap 90, window 30 (base features) | 30 | 90.9 | 89.7 | 1.2 [-0.5, 2.6] | 249 |
| A3 | cap 90, window 30 + hi_consistent+regime_onehot vs cap 90, window 30 (base features) | 40 | 99.7 | 99.2 | 0.4 [0.0, 0.9] | 249 |

A4 — subpopulation from the first N cycles (logistic regression, AUC [95% CI], strong = CI lower bound ≥ 0.80; subpopulation sizes {'0': 148, '1': 101}):

| N cycles | AUC [95% CI] | per repeat | n engines | strong |
|---|---|---|---|---|
| 30 | 0.995 [0.989, 0.999] | 0.994, 0.993, 0.995 | 249 | yes |
| 50 | 0.997 [0.992, 1.000] | 0.996, 0.996, 0.997 | 249 | yes |
| 100 | 1.000 [1.000, 1.000] | 1.000, 1.000, 1.000 | 249 | yes |

Winners: A1: cap 90; A2: cap 90, window 30; A3: cap 90, window 30 + hi_consistent+regime_onehot; stage_A: cap 90, window 30 + hi_consistent+regime_onehot
- A3 sensors_plus [s6, s10, s16] is infeasible as pre-registered: s16 is constant within regime(s) [0, 1, 2, 5] on the training data, so within-regime normalization is undefined (audit §E classed these sensors informative — contradiction flagged). Not adopted.
- DEVIATION (informational only, not eligible for adoption): sensors_plus_fittable = [s6, s10] reported below.
