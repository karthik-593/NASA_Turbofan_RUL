### Stage A — FD002

| config | critical RMSE [95% CI] | n engines | subpop 0 | subpop 1 | wall (min) |
|---|---|---|---|---|---|
| `A/FD002/cap090` | 5.70 [5.37, 6.07] | 260 | 5.75 [5.31, 6.25] (n=183) | 5.57 [5.12, 6.03] (n=77) | 18.8 |
| `A/FD002/cap105` | 5.76 [5.43, 6.13] | 260 | 5.79 [5.37, 6.27] (n=183) | 5.68 [5.16, 6.17] (n=77) | 18.6 |
| `A/FD002/cap125` | 5.93 [5.58, 6.32] | 260 | 5.91 [5.49, 6.38] (n=183) | 5.96 [5.34, 6.60] (n=77) | 15.5 |
| `A/FD002/cap140` | 6.03 [5.68, 6.43] | 260 | 6.00 [5.57, 6.47] (n=183) | 6.11 [5.38, 6.83] (n=77) | 18.8 |
| `A/FD002/cap090_w30` | 4.77 [4.53, 5.04] | 260 | 4.79 [4.50, 5.13] (n=183) | 4.71 [4.28, 5.15] (n=77) | 19.0 |
| `A/FD002/cap090_w45` | 4.14 [3.90, 4.41] | 260 | 4.12 [3.82, 4.43] (n=183) | 4.21 [3.79, 4.62] (n=77) | 19.0 |
| `A/FD002/cap090_w45_hi_consistent` | 3.92 [3.63, 4.21] | 260 | 3.79 [3.51, 4.10] (n=183) | 4.22 [3.64, 4.81] (n=77) | 12.5 |
| `A/FD002/cap090_w45_hi_pooled` | 3.96 [3.72, 4.23] | 260 | 3.69 [3.46, 3.95] (n=183) | 4.56 [4.04, 5.10] (n=77) | 19.9 |
| `A/FD002/cap090_w45_regime_onehot` | 4.13 [3.91, 4.39] | 260 | 4.08 [3.80, 4.39] (n=183) | 4.25 [3.83, 4.68] (n=77) | 19.1 |
| `A/FD002/cap090_w45_sensors_plus_fittable` | 4.10 [3.87, 4.35] | 260 | 4.03 [3.77, 4.32] (n=183) | 4.25 [3.84, 4.67] (n=77) | 20.6 |
| `A/FD002/cap090_w45_combined` | 3.92 [3.63, 4.21] | 260 | 3.79 [3.51, 4.10] (n=183) | 4.22 [3.64, 4.81] (n=77) | 21.1 |

Paired comparisons (Δ = challenger − incumbent, critical RMSE; better = CI entirely < 0):

| step | challenger | incumbent | Δ [95% CI] | engines challenger better | Wilcoxon p (n splits) | verdict |
|---|---|---|---|---|---|---|
| A1 | cap 90 | cap 125 | -0.23 [-0.34, -0.14] | 67% of 260 | 0.00031 (15) | **better** |
| A1 | cap 105 | cap 125 | -0.17 [-0.25, -0.10] | 62% of 260 | 0.0026 (15) | **better** |
| A1 | cap 140 | cap 125 | 0.10 [0.05, 0.16] | 40% of 260 | 0.0043 (15) | worse |
| A2 | cap 90, window 30 | cap 90, window 20 | -0.93 [-1.18, -0.65] | 64% of 260 | 6.1e-05 (15) | **better** |
| A2 | cap 90, window 45 | cap 90, window 20 | -1.55 [-1.92, -1.16] | 75% of 260 | 6.1e-05 (15) | **better** |
| A3 | cap 90, window 45 + hi_consistent | cap 90, window 45 (base features) | -0.23 [-0.38, -0.07] | 66% of 260 | 0.0034 (15) | **better** |
| A3 | cap 90, window 45 + hi_pooled | cap 90, window 45 (base features) | -0.18 [-0.33, -0.03] | 62% of 260 | 0.0043 (15) | **better** |
| A3 | cap 90, window 45 + regime_onehot | cap 90, window 45 (base features) | -0.01 [-0.05, 0.03] | 48% of 260 | 0.56 (15) | not different |
| A3-informational | cap 90, window 45 + sensors_plus_fittable (deviation) | cap 90, window 45 (base features) | -0.04 [-0.09, 0.00] | 53% of 260 | 0.11 (15) | not different (informational, not eligible) |
| A3-combined | cap 90, window 45 + hi_consistent | cap 90, window 45 (base features) | -0.23 [-0.37, -0.08] | 66% of 260 | 0.0034 (15) | **better** |

Decision check for each adopted change (caught % with lead ≥ 20 at matched mean wasted life; Δ = challenger − incumbent, positive = challenger catches more):

| step | challenger vs incumbent | budget (cycles) | caught % challenger | caught % incumbent | Δ [95% CI] | n engines |
|---|---|---|---|---|---|---|
| A1 | cap 90 vs cap 125 | 20 | 48.1 | 46.6 | 1.5 [-0.3, 3.5] | 260 |
| A1 | cap 90 vs cap 125 | 30 | 92.3 | 90.4 | 1.9 [0.4, 3.3] | 260 |
| A1 | cap 90 vs cap 125 | 40 | 99.4 | 99.2 | 0.2 [-0.6, 0.9] | 260 |
| A2 | cap 90, window 45 vs cap 90, window 20 | 20 | 49.6 | 48.1 | 1.5 [-2.2, 5.2] | 260 |
| A2 | cap 90, window 45 vs cap 90, window 20 | 30 | 96.7 | 92.3 | 4.5 [1.3, 7.2] | 260 |
| A2 | cap 90, window 45 vs cap 90, window 20 | 40 | 100.0 | 99.4 | 0.6 [0.0, 1.3] | 260 |
| A3 | cap 90, window 45 + hi_consistent vs cap 90, window 45 (base features) | 20 | 50.9 | 49.6 | 1.4 [-1.2, 4.2] | 260 |
| A3 | cap 90, window 45 + hi_consistent vs cap 90, window 45 (base features) | 30 | 99.4 | 96.7 | 2.7 [1.3, 4.3] | 260 |
| A3 | cap 90, window 45 + hi_consistent vs cap 90, window 45 (base features) | 40 | 100.0 | 100.0 | 0.0 [-0.0, 0.1] | 260 |

A4 — subpopulation from the first N cycles (logistic regression, AUC [95% CI], strong = CI lower bound ≥ 0.80; subpopulation sizes {'0': 183, '1': 77}):

| N cycles | AUC [95% CI] | per repeat | n engines | strong |
|---|---|---|---|---|
| 30 | 0.523 [0.445, 0.597] | 0.491, 0.507, 0.559 | 260 | no |
| 50 | 0.589 [0.509, 0.663] | 0.578, 0.588, 0.597 | 260 | no |
| 100 | 0.951 [0.922, 0.973] | 0.946, 0.945, 0.953 | 260 | yes |

Winners: A1: cap 90; A2: cap 90, window 45; A3: cap 90, window 45 + hi_consistent; stage_A: cap 90, window 45 + hi_consistent
- A3 sensors_plus [s6, s10, s16] is infeasible as pre-registered: s10 is constant within regime(s) [1]; s16 is constant within regime(s) [0, 1, 2, 4, 5] on the training data, so within-regime normalization is undefined (audit §E classed these sensors informative — contradiction flagged). Not adopted.
- DEVIATION (informational only, not eligible for adoption): sensors_plus_fittable = [s6] reported below.
- A3-combined: both health indices adopted; hi_consistent (lower point estimate) is the one combined, as the two share the same three columns
