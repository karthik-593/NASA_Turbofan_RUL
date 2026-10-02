# Artifact Inventory — `models/`

Generated 2026-10-02 by loading every file under `models/` (nothing deleted or modified).
`models/` is untracked in git, so the file timestamps (DD-MM-2026, local time) and contents are
the only provenance available.

## Summary

| Group | Files | Producing code in `src/`? | Producing code anywhere in repo? |
|---|---|---|---|
| A. Current bundles | `models/FD001/lstm/<version>/` ×2 | Yes — `training/train.py` | — |
| B. Notebook-02 tuned XGBoost (42 features) | `xgboost_FD00{1-4}.pkl`, `features_FD00{1-4}.joblib` | No | Yes — `notebooks/archive/02_modeling.ipynb` cells 07, 13 |
| C. Pre-refactor 258-feature pipeline | `xgboost_FD001_{base,asym,quantile,healthgate}.pkl`, `features_FD001_{asym,base}.joblib`, `lstm_FD00{1-4}.pt`, `comparison_results.joblib`, `hardened_all_results.joblib`, `cv_results_FD001.joblib` | No | **No** — no file in the working tree or git history (`git log --all -S`) writes them |

Group C comes from a feature pipeline and model wrapper that no longer exist: the pickles
reference the removed class `turbofan.features.engineering.FeatureArtifacts` and attributes the
current `XGBoostRUL` no longer has (`asymmetric_alpha`, `quantile_models_`,
`health_classifier_`, `apply_monotone_constraints`). The objects load only because pickle
restores attributes without calling `__init__`; none can be re-created from current code.

**Every metric stored in these artifacts was computed on the NASA test set**, from a single run
with no seed variation. They are listed for traceability only and cannot be used as selection
evidence.

## A. Current training bundles (`training/train.py`)

| Path | Contents |
|---|---|
| `FD001/lstm/20260603T142540Z/` | Pre-schema bundle: manifest has no `bundle_schema`, `config`, `seed` or `lib_versions`. `model.bin` = `{cfg, state}`; cfg hidden 32, 2 layers, dropout 0.3, lr 1e-3, batch 256, patience 10, seed 42. `feature_state.pkl` = global z-score stats, 14 sensors. Test metrics: crit RMSE 3.1284, global 17.5592, NASA 896.79, late 62%, n=100. |
| `FD001/lstm/20260603T183633Z/` | Current schema (`bundle_schema: 1`). Same cfg, same feature stats and same test metrics as above (crit 3.1284, n=100), plus `val_loss` 175.46 and lib versions (torch 2.11.0, xgboost 3.2.0, sklearn 1.8.0). This is what `version="latest"` resolves to. |

Neither manifest records a git SHA (addressed in a later commit).

## B. Notebook-02 artifacts (42-feature lean set)

Written by `notebooks/archive/02_modeling.ipynb` (`model.save(MODELS / f'xgboost_{name}.pkl')`,
`joblib.dump(stats, ...)`). Loadable with current `XGBoostRUL`.

| File | Type | Contents |
|---|---|---|
| `xgboost_FD001.pkl` | `XGBoostRUL` | 42 features; Optuna best params: depth 3, lr 0.0533, subsample 0.763, colsample 0.698, min_child_weight 6, alpha 0.449, lambda 2.304; n_estimators 1000, best_iteration 500, device cuda |
| `xgboost_FD002.pkl` | `XGBoostRUL` | 42 features; depth 5, lr 0.0352; best_iteration 483 |
| `xgboost_FD003.pkl` | `XGBoostRUL` | 42 features; depth 7, lr 0.0369; best_iteration 188 |
| `xgboost_FD004.pkl` | `XGBoostRUL` | 42 features; depth 7, lr 0.0449; best_iteration 296 |
| `features_FD00{1,3}.joblib` | dict | Global stats `{multi: False, s_mean, s_std}`, 14 sensors. FD001 values identical to the current bundle's `feature_state.pkl`. |
| `features_FD00{2,4}.joblib` | dict | `{multi: True, op_sc, km, s_mean, s_std}`; KMeans k=6, n_init 10, random_state 0; fit on 43,103 (FD002) / 49,072 (FD004) train rows; inertia 0.189 / 0.214 |

Stored metrics: none. The test scores for these models are printed in nb02 cell-14
(crit RMSE 8.96 / 5.68 / 4.59 / 8.64).

Note: `best_iteration` 500 for FD001 is exactly half of n_estimators 1000, which is unusual;
not investigated.

## C. Pre-refactor 258-feature pipeline

### Feature definition (from `feature_names_` in the XGBoost pickles)

258 features: for each of the same 14 sensors (`sensor_N` naming) — raw value; rolling
mean/std/min/max over windows 5, 15, 30; slope over 15 and 30; first difference; EWM span 15;
`tst` (unknown — likely time-since-trend) → 18 × 14 = 252; plus `cycle_raw`, `cycle_norm`,
three sensor interactions (`11×4`, `11×12`, `4×12`) and a PCA-1 `health_index`.

This is the feature family that `docs/problem_framing.md` and decision D09 describe as
"deliberately not included".

### `features_FD001_{asym,base}.joblib`

Byte-identical files. Instance of the removed `FeatureArtifacts` class (loaded with a stand-in
class). Fields: `dataset_name` FD001, `is_multi_regime` False, `op_scaler`, `km_model` None,
`cluster_means`/`cluster_stds` (1×14), `kept_sensors` (same 14), `pca_model` PCA(n_components=1),
`max_train_cycle` 362.0. Max lifetime of FD001 train is 362 (nb01 cell-6) and the means differ
from the current 80-engine-train stats (s2 642.6814 vs 642.6868), so these stats were likely fit
on all 100 train engines including validation — not verified.

### `xgboost_FD001_{base,asym,quantile,healthgate}.pkl`

All `XGBoostRUL` (old version), 258 features, FD001 only.

| File | Model | Key settings |
|---|---|---|
| `base` | XGBRegressor, squared error | depth 4, lr 0.0323, 253 trees, best_iteration 251 |
| `asym` | XGBRegressor, custom objective (`_asymmetric_mse_sentinel_`) | `asymmetric_alpha` 3.75; depth 4, lr 0.0195, 469 trees, best_iteration 412. Separate hyperparameters from `base`. |
| `quantile` | `base` + `quantile_models_` {0.1, 0.5, 0.9} | three XGBRegressor `reg:quantileerror`, 253 trees each |
| `healthgate` | `base` + `health_classifier_` | XGBClassifier `binary:logistic`, 253 trees, depth 4, on all 258 features |

No stored metrics in any of the four. `base` matches the `xgb` / FD001 entry of
`comparison_results.joblib` (same best_iter 251).

### `lstm_FD00{1-4}.pt`

`dict{model_state, model_config, history, best_val_rmse, best_epoch}`. Old architecture: 255
input features, hidden 64, 2 layers, dropout 0.3, head 64→32→1 (117,569 parameters — matches
`n_params` in `comparison_results.joblib`). Not loadable by the current `LSTMRUL` (different keys
and shapes).

| Dataset | best_val_rmse | best_epoch | epochs run |
|---|---:|---:|---:|
| FD001 | 17.98 | 12 | 20 |
| FD002 | 17.68 | 4 | 12 |
| FD003 | 14.30 | 8 | 16 |
| FD004 | 19.42 | 12 | 20 |

Validation definition unknown (no producing code).

### `comparison_results.joblib`

`dict{xgb, lstm}` → per dataset: `test_rmse, test_mae, test_score, bucket (DataFrame), train_time,
y_test, y_pred, best_iter | best_epoch, n_params`. Per-test-engine predictions are stored, so the
bucket numbers below were re-computed from `y_test`/`y_pred` and match the stored `bucket`
tables.

| Model | Dataset | n test | n crit | crit RMSE | global RMSE | NASA score |
|---|---|---:|---:|---:|---:|---:|
| xgb (258 feat) | FD001 | 100 | 19 | 3.68 | 12.99 | 244 |
| xgb | FD002 | 259 | 58 | 4.12 | 25.33 | 6051 |
| xgb | FD003 | 100 | 15 | 4.22 | 14.93 | 435 |
| xgb | FD004 | 248 | 49 | 7.46 | 26.78 | 5080 |
| lstm (255 feat) | FD001 | 100 | 19 | 17.91 | 23.53 | 1692 |
| lstm | FD002 | 259 | 58 | 10.44 | 28.63 | 9498 |
| lstm | FD003 | 100 | 15 | 14.96 | 23.35 | 2772 |
| lstm | FD004 | 248 | 49 | 12.42 | 29.98 | 10920 |

### `hardened_all_results.joblib`

`dict[dataset]{sym, hard}` → `{rmse, crit, S, late, ci}`. "sym" = symmetric loss; "hard" is
presumably the asymmetric/"hardened" variant (no code to confirm). `ci` is an interval on global
RMSE; method and n not recorded. FD003 `sym` is identical to `comparison_results` xgb FD003, so
`sym` is the 258-feature XGBoost.

| Dataset | Variant | global RMSE | ci | crit RMSE | NASA S | late % |
|---|---|---:|---|---:|---:|---:|
| FD001 | sym | 13.36 | [11.06, 15.63] | 3.67 | 273 | 60.0 |
| FD001 | hard | 14.34 | [11.54, 17.15] | 3.83 | 428 | 48.0 |
| FD002 | sym | 25.46 | [22.25, 28.71] | 4.32 | 6381 | 42.1 |
| FD002 | hard | 25.56 | [22.40, 28.86] | 3.64 | 8068 | 38.6 |
| FD003 | sym | 14.93 | [12.25, 17.42] | 4.22 | 435 | 57.0 |
| FD003 | hard | 15.05 | [12.34, 17.50] | 2.59 | 417 | 51.0 |
| FD004 | sym | 26.63 | [23.85, 29.37] | 6.58 | 5133 | 54.0 |
| FD004 | hard | 26.61 | [23.94, 29.45] | 6.50 | 9114 | 52.4 |

This is the only artifact in the repo with any interval estimate, and the CI covers global RMSE,
not the headline metric.

### `cv_results_FD001.joblib`

DataFrame, 5 folds + mean/std rows; columns rmse, mae, cmapss_score, late_pct, n_engines,
critical_rmse, urgent/monitor/healthy_rmse, fold. Each fold has n_engines = 20 (5-fold CV over the
100 FD001 train engines). Only `critical_rmse` is populated, equal to `rmse` in every fold, with
late_pct 95–100%. This is consistent with scoring each run-to-failure validation engine at its
**last cycle, where true RUL is always 0** — so the reported crit RMSE 2.94 ± 0.61 measures error
at failure only, not RUL estimation across the range. Model and features not recorded.

## Findings to review

1. **Contradiction with D07/D09 (lean 42-feature set).** On the test set, the 258-feature
   XGBoost scored crit RMSE 3.68 / 4.12 / 4.22 / 7.46 (FD001–FD004). The 42-feature tuned
   XGBoost (nb02) scored 8.96 / 5.68 / 4.59 / 8.64. The shipped LSTM's 5-seed means
   (`docs/results.md`) are 3.14 / 5.04 / 3.77 / 6.64, so on FD002 the old XGBoost is lower.
   All single-run, test-set, no CI, critical n = 15–58 — not admissible, but enough to say the
   decision to drop these feature families was not shown to be harmless. Logged as X06 in
   `docs/decisions.md`.
2. **The asymmetric-loss experiment was already run** (`asym` pickle, `hardened_all_results`).
   nb02 cell-15 heads it "Future experiment (not yet run)" while also citing "an earlier run";
   `PROJECT_SNAPSHOT.md` says "scoped but not implemented". Its results exist only in these
   untracked artifacts. The stored results show lower late % on all four datasets at the cost of a
   higher NASA score on 3 of 4.
3. **No artifact in groups B or C can be reproduced from the current code.** Group C has no
   producing code at all.
4. Old normalization stats appear to have been fit on all FD001 train engines including those
   later used for validation (`max_train_cycle` 362, differing means). Not verified.
5. `cv_results_FD001.joblib` evaluates only at RUL = 0 and should not be read as a cross-validated
   RUL error.
