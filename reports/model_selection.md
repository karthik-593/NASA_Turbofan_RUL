# Model selection under protocol v2 — pre-registration and results (D51)

**Pre-registered 2026-10-03 at commit `2b512d4`, before any selection run.** The rules, grids,
feature definitions and thresholds below are fixed. Any later deviation is written into the
stage summary as a deviation, with its reason. Stage summaries are appended below the
pre-registration as each stage completes.

## 0. Pre-registration

### Scope

- Training files only (`load_train`). The NASA test set stays sealed: `final_eval` is not run in
  this work (rule 3). The only test-side input remains the benchmark view's label histogram (D46).
- Every dataset (FD001–FD004) is selected separately. Results are never pooled across datasets.
- Every run goes through the pipeline entry point (`python -m turbofan.pipeline …`), so resume
  (D49) and the GPU-teardown workaround (D50) apply, and every run is logged to MLflow.

### Ranking rule

- **Metric:** deployment-view critical-bucket RMSE against uncapped truth (`critical_rmse`,
  D18/D46). It is computed on CV held-out engines over 5 folds × 3 repeats (`params.yaml` `cv`),
  with a 1,000-replicate engine bootstrap.
- **"Better":** the paired Δ (challenger − incumbent) has a 95% paired engine-bootstrap CI
  entirely below 0 (`evaluation.compare.paired_compare`: same engines, cycles, folds and seeds).
- **Ties:** if the CI includes 0, the **simpler / cheaper option wins**. Within each stage,
  "simpler" is defined as follows:
  - the incumbent (current default) over a change;
  - fewer features over more;
  - a smaller rolling window or sequence length over a larger one;
  - XGBoost over the LSTM (cheaper to train and to serve);
  - untuned over tuned.
- **Caps:** candidates with different `rul_cap` are compared only on cap-invariant metrics.
  `compare.CAP_INVARIANT` refuses anything else (D01).
- **Decision check:** every finalist gets a decision check (D47): caught % with lead ≥ 20 cycles
  at matched mean-wasted-life budgets of 20 / 30 / 40 cycles. It is reported paired, with
  paired engine-bootstrap CIs (`compare.matched_budget_compare`).
  - **"Better on the decision check"** means the paired Δ caught % CI is entirely above 0 at
    ≥ 1 budget, and entirely below 0 at none.
- **Two-candidate rule:** if, for a dataset, the RMSE-better finalist is **not** better on the
  decision check, both of its top-2 finalists go forward as **locked-pending**. The final pick
  then happens in the decision-layer stage, on expected cost with calibrated intervals.
- **Multiplicity:** no correction across the many comparisons of Stages A–C. Some screening
  adoptions may be false positives. Stage D (5 seeds) is the confirmation, and only its
  comparisons decide what is locked.
- **Design:** one factor at a time. Interactions between factors are **not tested** (a
  limitation). The single exception is the pre-registered A3-combined run below.

### Stage A — screening with XGBoost

Setup:

- XGBoost at registry defaults (`build_registry`), seed 42, 5 × 3 folds, every dataset.
- Base features: the 14 `sensors.use` sensors × {within-regime z, rolling mean, rolling slope}.
- **Decision check (informational):** every adopted change is also reported with a decision
  check against its incumbent, and with critical RMSE per subpopulation (D35).

Sub-stages:

- **A1 — cap.** `rul_cap` ∈ {90, 105, 125, 140}; the incumbent is 125.
  - Each cap is compared with 125.
  - Among the caps that beat 125, the one with the lowest point estimate is adopted.
  - If none beats 125, 125 stays.
- **A2 — rolling window.** `window` ∈ {20, 30, 45}, at the A1 cap; the incumbent is 20.
  - Same rule as A1.
- **A3 — feature blocks.** Each block is added alone to the base set, at the A1 / A2 winners,
  and compared with base. The blocks:
  - **`hi_consistent`:** fitted per fold on the inner-training engines only.
    - For each base sensor, compute the per-engine Spearman ρ(within-regime z, uncapped time to
      failure).
    - A sensor is *direction-consistent* if ≥ 90% of engines share the majority sign. This is
      the audit §E rule.
    - Take PC1 of the within-regime z of the consistent sensors over the inner-training rows,
      oriented to rise toward failure.
    - Features: the PC1 score, plus its rolling mean and rolling slope over the run's window.
    - The fitted state is applied unchanged to inner-validation and held-out engines.
  - **`hi_pooled`:** the same construction, but over all base sensors (no consistency filter).
  - **`regime_onehot`** (FD002, FD004 only): one indicator column per fitted operating regime
    (k = 6, `n_regimes`).
  - **`sensors+`:** the sensors audit §E classes as informative but outside `sensors.use`, as
    one block per dataset, 3 features each like the base sensors:
    - FD001: s6
    - FD002: s6, s10, s16
    - FD003: s6, s10
    - FD004: s6, s10, s16
  - **Adoption:** a block is adopted if it beats base.
  - **A3-combined:** if ≥ 2 blocks are adopted, their union is run once as A3-combined. It is
    carried forward if it beats base. Otherwise the adopted block with the lowest point
    estimate is carried.
- **A4 — early identifiability of the subpopulation.** Can an engine's D35 subpopulation (a
  full-trajectory label) be predicted causally from its first N cycles, N ∈ {30, 50, 100}?
  - **Features:** per base sensor, the mean and OLS slope of the within-regime z over cycles
    1..N. Normalization is fitted on the fold's training engines.
  - **Classifier:** standardized logistic regression (L2, C = 1).
  - **Folds:** the same repeated StratifiedGroupKFold (5 × 3, stratified by the label).
  - **Metric:** ROC AUC of each engine's out-of-fold probability, averaged over the 3 repeats,
    with a 1,000-replicate engine-bootstrap CI.
  - **"Strong"** means the AUC CI lower bound is ≥ 0.80. A strong result is *reported* as a
    candidate feature for a later block test. It is not forced into Stage A.

### Stage B — LSTM (registry defaults, seed 42, at the Stage-A cap per dataset)

- **B1:** `seq_len` ∈ {20, 30, 45}; the incumbent is 30.
- **B2** (FD002, FD004): input channels base vs base + `regime_onehot`, at the B1 winner.
- The LSTM keeps the base sensor channels. The A3 blocks are XGBoost features and are not
  carried into LSTM channels.
- Same rule as Stage A.

### Stage C — tuning

- **Configs tuned:** the best XGBoost config and the best LSTM config per dataset after A / B.
- **Search:** Optuna TPE (seed 42), with 30 trials for XGBoost and 20 for the LSTM. Search
  spaces live in `params.yaml` `tuning`.
- **Objective:** deployment critical RMSE on the **repeat-1 folds only** (5 splits).
- **Evaluation:** the tuned config and its untuned incumbent are compared on **repeats 2–3**
  (10 splits) with the paired rule. The tuned config is adopted only if it beats the incumbent.
- **Residual optimism (stated, not removed):** repeats 2–3 reshuffle the *same* training
  engines that tuned the config on repeat 1. Every held-out engine in repeats 2–3 was a tuning
  engine in some repeat-1 fold, so the repeat-2/3 estimate is optimistic for the tuned config.

### Stage D — confirmation

- **Finalists ("top-2"):** per dataset, the best XGBoost config and the best LSTM config after
  Stage C.
- **Runs:** each finalist with all 5 seeds (`params.yaml` `seeds`) × 5 × 3 folds.
- **Reported:**
  - paired Δ critical RMSE with a CI;
  - Wilcoxon over fold × seed;
  - per-subpopulation critical RMSE;
  - the matched-budget decision check.
- **Outcome:** the ranking rule and the two-candidate rule produce either one **locked** spec
  or two **locked-pending** specs per dataset.

### Stage E — write-up

- **Outputs:**
  - `notebooks/04_model_selection.ipynb` (style guide in `notebooks/README.md`);
  - this report;
  - candidate specs as YAML in `specs/` (`evaluation.spec` format).
- **Specs:** `locked: true` only where the rules give a single winner. Otherwise
  `locked: false`, `status: locked-pending`.
- **D25 ("ship LSTM"):** the report says whether it survives.
- **Not run:** `final_eval`.

### Order and stopping

Stages run sequentially. Each stage is resumable and pushed with a short summary appended here.
**Work stops after Stage A for review** before Stage B starts.

---

## Stage A — screening results (XGBoost, seed 42, 5 × 3 folds)

Completed 2026-10-04. Full per-dataset tables (every configuration with CI, n and
per-subpopulation results; every paired comparison with its Wilcoxon p; decision checks; A4) are
in `reports/selection/stage_A_FD00x.md`, machine-readable in `.json`. Δ = challenger − incumbent
in deployment-view critical RMSE (cycles); "better" = paired engine-bootstrap 95% CI entirely
below 0.

### Outcome per dataset

| dataset | A1 cap | A2 window | A3 blocks adopted | Stage-A config | critical RMSE [95% CI] (n engines) | vs original default (cap 125, window 20, base) Δ [95% CI] |
|---|---|---|---|---|---|---|
| FD001 | 90 | 45 | health index (pooled) | cap 90, window 45, `hi_pooled` | 3.41 [3.05, 3.76] (100) | −1.79 [−2.25, −1.36] |
| FD002 | 90 | 45 | health index (consistent) | cap 90, window 45, `hi_consistent` | 3.92 [3.63, 4.21] (260) | −2.01 [−2.42, −1.63] |
| FD003 | 90 | 45 | health index (consistent) | cap 90, window 45, `hi_consistent` | 3.55 [3.12, 4.08] (100) | −1.44 [−2.11, −0.75] |
| FD004 | 90 | 30 | health index (consistent) + regime one-hot | cap 90, window 30, `hi_consistent` + `regime_onehot` | 5.98 [5.51, 6.47] (249) | −1.14 [−1.49, −0.82] |

The last column is one direct paired comparison per dataset added for context (not part of the
pre-registered chain, which tests one factor at a time). Original-default critical RMSE: FD001
5.19 [4.78, 5.64], FD002 5.93 [5.58, 6.32], FD003 4.99 [4.40, 5.61], FD004 7.12 [6.62, 7.64].

### What each step found

- **A1 cap.** 90 and 105 both beat 125 on every dataset (e.g. FD004 cap 90 −0.54 [−0.70, −0.38],
  n = 249); 140 is worse than 125 on every dataset. 90 has the lowest estimate everywhere.
  **Flag for review: 90 is the lowest value of the pre-registered grid, so the optimum may lie
  below it** — the grid did not test < 90.
- **A2 window.** 45 adopted on FD001/FD002/FD003 (e.g. FD002 −1.55 [−1.92, −1.16]); on FD004
  only 30 beats 20 (−0.40 [−0.73, −0.06]); 45 is not different there (−0.21 [−0.77, 0.36]).
  **Same flag: 45 is the grid's upper edge for three datasets.**
- **A3 blocks.** The direction-consistent health index beats base on all four datasets (FD001
  −0.47 [−0.84, −0.18], FD002 −0.23 [−0.38, −0.07], FD003 −0.25 [−0.43, −0.06], FD004
  −0.19 [−0.34, −0.04]). The pooled index also beats base on FD001 (−0.57 [−0.91, −0.24],
  the lower estimate, so FD001 carries `hi_pooled`) and FD002, and is not different on FD003 /
  FD004. Regime one-hot: barely better on FD004 (−0.08 [−0.17, −0.01]), not different on FD002.
  Extra sensors (`sensors+`): not different on FD001 / FD003; see deviations for FD002 / FD004.
- **Decision check** (caught % with lead ≥ 20 at matched wasted life): no adopted change is worse
  at any budget; several are better at the 30-cycle budget (e.g. FD002 window 45 +4.5
  [1.3, 7.2] pp; FD004 cap 90 +1.7 [0.2, 3.1] pp). Against the original default the Stage-A
  configs catch more at 30 cycles on FD002 (+9.0 [6.2, 12.0]), FD003 (+4.3 [0.9, 8.2]) and FD004
  (+5.9 [3.2, 8.7]); FD001 not different (+1.2 [−1.4, 5.2]).
- **A4 early identifiability** (AUC [95% CI], 3-repeat average):

  | N cycles | FD001 (60/40) | FD002 (183/77) | FD003 (56/44) | FD004 (148/101) |
  |---|---|---|---|---|
  | 30 | 0.66 [0.54, 0.77] | 0.52 [0.45, 0.60] | 0.997 [0.989, 1.000] | 0.995 [0.989, 0.999] |
  | 50 | 0.72 [0.62, 0.81] | 0.59 [0.51, 0.66] | 0.994 [0.983, 1.000] | 0.997 [0.992, 1.000] |
  | 100 | 0.95 [0.91, 0.99] | 0.95 [0.92, 0.97] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

  "Strong" (CI lower bound ≥ 0.80) from 30 cycles on FD003 and FD004 — the two datasets the
  C-MAPSS readme lists with two fault modes — and only from 100 cycles on FD001 / FD002. An
  exploratory check (not pre-registered, not MLflow-logged) on FD003 N = 30: with shuffled labels
  the AUC is 0.55 [0.43, 0.66], so the pipeline does not leak; the real separation comes from
  early *levels* (first-30-cycle mean z of s17, s3, s2, s11, s4 differ by ≈ 1.4 SD between the
  groups), not early slopes. Per the pre-registration this is **reported, not forced**: a
  causal subpopulation-probability feature is a candidate for a later block test on FD003/FD004.

### Deviations and incidents (all recorded, none changes a pre-registered rule)

1. **Resume-key bug (D49).** The resume key omitted the new `features` section, so the first A3
   runs on FD001 and FD003 silently reloaded their base runs (Δ exactly 0.00). Found from the
   results, fixed (`39fb46b`, with a test that every params section is classified), invalid runs
   deleted locally and tagged `invalid=resume_key_missing_features` in MLflow, A3 re-run. A1/A2
   were unaffected.
2. **`sensors+` infeasible on FD002 / FD004 (rule 9 — contradicts audit §E).** s10 (FD002
   regime 1) and s16 (FD002 5 of 6 regimes, FD004 4 of 6) are constant within operating regimes
   on the training data, so within-regime normalization is undefined; audit §E had classed them
   informative. The pre-registered block was recorded as infeasible and **not adopted**. The
   fittable subset ran as an informational deviation, **not eligible for adoption**: FD002 [s6]
   −0.04 [−0.09, 0.00] (not different); FD004 [s6, s10] +0.49 [0.03, 1.01] (worse).
3. **A3-combined reduces to one block when both health indices are adopted** (they share the
   same three columns): the union then holds the index with the lower estimate (FD001 pooled,
   FD002 consistent), identical to that block's own run.
4. **Orchestrator restarts.** FD002 / FD004 orchestrators hit a 2-hour process limit and were
   relaunched; finished configurations were reused, unfinished splits resumed from MLflow. No
   effect on results.

### Limitations carried forward

- One factor at a time: cap was chosen at window 20 and base features; window at the A1 cap;
  blocks at both. Interactions (e.g. cap × window) are untested.
- About 40 paired comparisons across the four datasets without multiplicity correction; the
  smallest adopted effects (FD004 regime one-hot, CI upper bound −0.01) are the most likely
  false positives. Stage D (5 seeds) is the confirmation.
- Single seed (42) throughout.
- Compute: 41 configuration runs, 9.8 h of summed run time (four datasets in parallel, so each run
  was slowed by CPU contention), ≈ 4.5 h elapsed including the reruns above.

### Carried into Stage B (pending review)

Per dataset: cap 90 for the LSTM (B1 seq_len, B2 regime one-hot channels on FD002/FD004), and
the Stage-A XGBoost configs above as the XGBoost candidates for Stage C. **Work stops here for
review before Stage B.**

---

## Addendum pre-registration (post-Stage-A, 2026-10-04)

**This section was written after the Stage A results were seen.** Its motivation is the Stage A
review: cap 90 and window 45 won at the edges of their grids, the `sensors+` block was
infeasible on two datasets, and A4 found the subpopulation identifiable early on FD003/FD004.
It fixes the follow-up rules and definitions below **before any of the runs it describes**.
The §0 rules (ranking metric, "better", ties to the simpler option, one factor at a time,
decision check, two-candidate rule, no multiplicity correction in screening) carry over
unchanged unless stated here.

### Changes since Stage A that affect every later run

- **Closed-form rolling slope (D52, `9e98ebe`).** It equals the former `np.polyfit` slope to
  1.8e-15 but is not bit-identical. So no new run is paired with a Stage A run: every paired
  comparison below re-runs its incumbent under the current code (tags `X/…`, `B/…`, `C/…`,
  `D/…`). The A1-check (below) is the one exception: it re-reads the Stage A A1 runs only.
- **Code freeze before Stage B.** The code for every stage (B–D orchestration, the new feature
  blocks, tuning) is committed before the first addendum run. Resume (D49) keys on the `src/`
  tree, so Stage D can then reuse the seed-42 splits of its finalists. Any later `src/` change
  is reported as a deviation.

### Known metric bias, and why the cap grid is not extended

Critical RMSE (deployment view) structurally favours a lower `rul_cap`: a narrower training
target puts more of the model's capacity on the low-RUL region the critical bucket scores. For
that reason Stage A's "cap 90 wins everywhere" is not taken at face value, and the cap is
**not extended below 90**.

### A1-check — is cap 90 an artefact of that bias? (no new runs)

- **Data:** the existing A1 runs, cap 90 vs cap 125, per dataset: XGBoost, base features,
  window 20, seed 42, 5 × 3 folds, deployment view. The held-out predictions are those logged
  to MLflow by the A1 parent runs (identical copies in `selection/A/<ds>/cap090|cap125/`).
- **Metrics,** each paired (cap 90 − cap 125) with a 1,000-replicate paired engine-bootstrap
  CI. Each is chosen because neither cap binds where it is measured:
  - **urgent-bucket RMSE** against uncapped truth (truth in [25, 50), below both caps);
  - **late %** in the critical and in the urgent bucket against uncapped truth (truth < 50).
    Overall late % favours the lower cap by construction (a capped model is "early" on every
    cycle above its cap), so it is reported for information only;
  - the **matched-budget decision check** (caught % with lead ≥ 20 at mean wasted life
    20 / 30 / 40 cycles; D47).
- **Rule:**
  - Cap 90 is **not worse** on a dataset when:
    - the urgent-RMSE Δ CI does not lie entirely above 0;
    - neither late-% Δ CI lies entirely above 0;
    - no decision-check Δ CI lies entirely below 0.
  - Cap 90 is **confirmed** only if it is not worse on every dataset.
  - Otherwise the addendum **stops**: the A1-check is reported and nothing below runs.
- To make these metrics comparable across caps, `compare.paired_compare` allows a bucket metric
  against uncapped truth when the bucket's upper bound is ≤ both caps (critical and urgent at
  caps ≥ 50). Every other cross-cap metric stays refused (D01).

### A2-ext — longer windows

- **Setup:** per dataset, window ∈ {60, 90} vs the A2 winner (FD001–FD003: 45, FD004: 30).
  All at cap 90 with base features, XGBoost, seed 42, 5 × 3 folds. The incumbent is re-run.
- **Rule:** same as A2. Among the windows that beat the incumbent, the one with the lowest
  point estimate is adopted.
- **Stopping:** if 90 wins, 90 is carried forward and the edge is documented as a limitation.
  There is no further extension.
- **Follow-on:** if the window changes on a dataset, A3 is re-run there at the new window: the
  same blocks, the same rule and the same A3-combined step, with `sensors+` only where it is
  feasible (FD001, FD003). Its result replaces the Stage A A3 result for that dataset.

### A5a — baseline-deviation block (all datasets)

- **Features,** per base sensor, from its within-regime z (normalization fitted per fold, as
  for every block):
  - `<s>_base`: the mean of z over the engine's cycles 1..min(t, 30). It is an expanding mean
    until cycle 30, then fixed. Causal: at cycle t it uses cycles ≤ t only.
  - `<s>_dev`: `<s>_mean` (the run's rolling mean) − `<s>_base`.
  - 28 columns. The number of baseline cycles is `features.baseline_cycles` = 30.
- **History requirement:** an engine whose first observed cycle is not 1 raises an error,
  rather than silently using a shifted baseline.
  - **Serving requirement (documented, not built):** serving needs the asset's full history
    from cycle 1, or a per-asset baseline stored once at cycle 30. Today's `/predict` takes a
    recent window only, and `save_bundle` refuses any active feature block.
- **Comparison:** the current adopted spec (after A2-ext / A3) + baseline vs that spec,
  re-run. Adopted if better.

### A4 re-run — through the pipeline, with a label-shuffle control

- **Re-run:** the `identifiability` stage, unchanged except for the control. Logged to
  MLflow, per dataset.
- **Control:** for each N, labels are permuted across engines (seed `selection.seed`), and the
  identical procedure gives a shuffled-label AUC with an engine-bootstrap CI. This logs and
  replaces the exploratory FD003 check of the Stage A summary.
- **Gate for A5b,** per dataset, all three conditions required:
  - real AUC ≥ 0.99 at N = 30;
  - real CI lower bound ≥ `auc_strong_lower` (0.80);
  - shuffled-label CI contains 0.5.

### A5b — subpopulation-probability block (FD003, FD004 only, if the gate passes)

- **Labels:** in every fold, the D35 subpopulation labels are re-derived on the fold's
  inner-training engines only (the `analysis.subpopulation` clustering on their full
  trajectories). No held-out or inner-validation engine shapes them.
- **Classifier (the A4 model):**
  - inputs: the mean and OLS slope of each base sensor's within-regime z over cycles 1..30
    (`features.subpop_prob_cycles`);
  - model: standardized L2 logistic regression with C = `features.subpop_prob_c` (1.0, as in
    A4), fitted on the inner-training engines.
- **Feature:** `subpop_p` = the predicted probability of label 1. It is defined from cycle 30
  on; before cycle 30 it is missing (NaN), handled by XGBoost's native missing-value split.
  - **Inner-training rows** get cross-fitted probabilities: 5 stratified inner folds
    (`features.subpop_prob_inner_folds`, seed `features.subpop_prob_seed` = 0). The training
    feature therefore carries out-of-sample error, like at serving.
  - **Inner-validation and held-out engines** get the classifier fitted on all inner-training
    engines.
- **Scope and constraints:**
  - XGBoost only, because the LSTM channels cannot carry a missing value. A sequence model with
    this block raises an error.
  - The serving requirement is as for A5a (cycles 1..30 must be seen).
  - D35 is updated: the label stays a training target only; this causal derived feature is
    permitted because A4 shows it is predictable from the first 30 cycles.
- **Comparison:** the spec after A5a + `subpop_p` vs the spec after A5a. Adopted if better.

### Audit correction (recorded in `docs/decisions.md`)

Audit §E classed a sensor "constant" only when it was single-valued within *every* regime, and
reported its within-regime std pooled across regimes. That masked per-regime constancy: s10
(FD002 regime 1) and s16 (FD002 5 of 6 regimes, FD004 4 of 6) are constant within some
regimes, so within-regime normalization is undefined for them. `sensors+` is infeasible on
FD002/FD004 and is not adopted (Stage A, deviation 2).

### Stage B — LSTM (replaces the §0 Stage B text)

- **Setup:** registry defaults, seed 42, 5 × 3 folds, cap 90 on every dataset (cap 90 having
  passed the A1-check).
  - The run's `window` is set to the dataset's adopted XGBoost window. It matters only for the
    `<s>_dev` channels.
- **B1 — sequence length:** `seq_len` ∈ {20, 30, 45, 60}, with base channels (14 within-regime
  z + mask); the incumbent is 30.
  - If 60 wins, it is carried forward and flagged as a grid edge, with no extension.
- **B2 — channels,** each added alone to base at the B1 winner:
  - `regime_onehot` (FD002 / FD004);
  - the health-index score `hi` of the variant XGBoost adopted on that dataset (if any);
  - the baseline-deviation channels `<s>_base`, `<s>_dev` (only where A5a was adopted).
- **Adoption:** a channel block is adopted if it beats base. If ≥ 2 are adopted, their union
  is run once (B2-combined) and carried if it beats base; otherwise the adopted block with the
  lowest estimate is carried.
- **Same rules** as Stage A, including the informational decision check for every adopted
  change.

### Stage C — tuning (as §0, with the search spaces fixed here)

- **Configs tuned:** per dataset, the final XGBoost spec (after the addendum) and the final
  LSTM spec (after B).
- **Search:** Optuna TPE (seed 42), 30 trials for XGBoost and 20 for the LSTM.
  - Each trial is one pipeline run (`cv_select`) on the repeat-1 folds: `cv.n_repeats` = 1 with
    the same `cv.seed` reproduces repeat 1 of the 5 × 3 scheme exactly.
  - Seed 42 throughout.
  - **Objective:** the deployment critical RMSE point estimate over those 5 splits. Trials run
    with `bootstrap.n_boot` = 200, since their CIs are not used.
- **Search spaces** (`params.yaml` `tuning`):
  - **XGBoost** (`models.xgboost.params`):
    - `max_depth` int [3, 10];
    - `learning_rate` log [0.01, 0.3];
    - `subsample` [0.5, 1.0];
    - `colsample_bytree` [0.5, 1.0];
    - `min_child_weight` log [1, 50];
    - `reg_alpha` log [1e-3, 10];
    - `reg_lambda` log [1e-3, 10];
    - `n_estimators` fixed at 2,000 for tuned configs (early stopping at 50 rounds decides).
  - **LSTM** (`models.lstm`):
    - `hidden` ∈ {16, 32, 64, 128};
    - `layers` int [1, 3];
    - `dropout` [0.0, 0.5];
    - `lr` log [3e-4, 3e-3];
    - `batch_size` ∈ {128, 256, 512};
    - `head_hidden` ∈ {8, 16, 32}.
- **Evaluation:**
  - The best trial's parameters run on the full 5 × 3 folds (seed 42).
  - They are compared with the untuned incumbent on **repeats 2–3 only** (10 splits; both
    runs' repeat-1 points are dropped), with the paired rule. Adopted only if better.
  - Residual optimism as stated in §0.
- **Resume:** a restarted study replays its trials deterministically (same sampler seed, same
  history), and completed trial runs are reused from `selection/`.

### Stage D — confirmation (replaces the §0 Stage D text)

- **Finalists:** per dataset, the best XGBoost spec and the best LSTM spec after Stage C. Each
  runs with all 5 seeds (`params.yaml` `seeds`) × 5 × 3 folds.
- **Leave-one-out check of every adopted change** in each finalist spec: the spec vs the spec
  minus that change, also with 5 seeds. "Minus" reverts to the original default:
  - cap 90 → 125;
  - window → 20 (XGBoost);
  - `seq_len` → 30 (LSTM);
  - the block → removed;
  - tuned parameters → registry defaults.
- **Parsimony:** a change is kept only if spec − (spec minus change) has a paired Δ critical
  RMSE CI entirely below 0. Otherwise it is **dropped**.
  - **Cap** is kept only if, in addition, the A1-check criteria (urgent RMSE, critical and
    urgent late %, decision check) are not worse at 5 seeds.
  - If exactly one change is dropped, that leave-one-out run becomes the finalist.
  - If several are dropped, the spec without all of them is run (5 seeds) and becomes the
    finalist, unless the full spec beats it. That case is reported as an interaction.
- **Finalist comparison** (XGBoost vs LSTM), on paired Δ critical RMSE (5 seeds × 15 splits):
  - "Better" needs the CI to exclude 0. A tie goes to XGBoost (simpler and cheaper).
  - Also reported: Wilcoxon over fold × seed, per-subpopulation critical RMSE, and the
    matched-budget decision check between the finalists.
- **Locking:**
  - **RMSE-better winner:** locked if it is also better on the decision check. Otherwise both
    finalists are locked-pending.
  - **Tie:** XGBoost wins and is locked, unless the LSTM is better on the decision check; then
    both are locked-pending.

### Stage E (as §0)

- **Outputs:**
  - `notebooks/04_model_selection.ipynb`;
  - this report;
  - candidate specs in `specs/<dataset>_<model>.yaml` (`evaluation.spec` fields plus the
    feature-block, window / `seq_len` and tuned-parameter overrides);
  - `locked: true` only for a single winner, otherwise `locked: false`,
    `status: locked-pending`.
- **Not run:** `final_eval`. `params.yaml` defaults are not changed (a later step, after
  review).

### Order, stopping and compute

- **Order:** A1-check → A2-ext (→ A3 re-run) → A5a → A4 → A5b → B → C → D → E. Datasets run
  serially.
- **Push:** after each stage, with a summary appended here.
- **Stops:** only if the A1-check fails, or after Stage E.
- **Estimate** (measured 2026-10-04, current code, one fold of FD001 / FD004):
  - XGBoost: 1.8 / 3.6 s per split, i.e. ~2–2.5 min per 5 × 3 configuration including
    evaluation.
  - LSTM (`seq_len` 30): 9.5 / 41 s per split, i.e. ~4 / ~12 min per configuration.

  | stage | runs | estimate |
  |---|---|---|
  | Addendum | ~8–13 XGBoost configurations per dataset | ~1.5–2 h |
  | B | ~7–8 LSTM configurations per dataset, `seq_len` up to 60 | ~5 h |
  | C | 120 XGBoost + 80 LSTM trials on 5 splits, plus 8 full evaluations | ~7 h |
  | D | 2 finalists + ~9 leave-one-out specs per dataset, 5 seeds (the LSTM on FD002/FD004 dominates) | ~12 h |
  | E | notebook and report | <1 h |

  ≈ 26 h serial in total.

---

## Addendum results (A1-check and Stage X, XGBoost, seed 42, 5 × 3 folds)

Completed 2026-10-04 at code `521ada5`.

- **Full tables:**
  - `reports/selection/a1_check.md`;
  - `reports/selection/stage_X_FD00x.md` / `.json`.
- **Re-runs:** every incumbent was re-run under the current code (D52).
- **Δ:** challenger − incumbent, critical RMSE, paired engine-bootstrap 95% CI.

**A1-check — cap 90 confirmed** on every dataset (Δ = cap 90 − cap 125):

- **Urgent-bucket RMSE:** better everywhere. FD001 −2.45 [−3.32, −1.61], FD002 −3.35
  [−3.95, −2.78], FD003 −2.12 [−3.09, −1.18], FD004 −2.75 [−3.37, −2.16].
- **Critical late %:** lower everywhere (−2.2 to −3.6 pp).
- **Urgent late %:** not different, and lower on FD003.
- **Decision check:** worse at no budget, and better at 30 cycles on FD002 (+1.9 [0.6, 3.5])
  and FD004 (+1.7 [0.3, 3.1]).

So cap 90 is not an artefact of the critical-RMSE bias on these metrics.

**Stage X outcome: no change to any Stage A spec.**

| dataset | A2-ext: window 60 vs winner | window 90 vs winner | A5a + baseline | A5b + subpop_prob | carried spec |
|---|---|---|---|---|---|
| FD001 | −0.21 [−0.46, 0.05] | +0.16 [−0.27, 0.55] | −0.01 [−0.21, 0.22] | out of scope | cap 90, window 45, `hi_pooled` |
| FD002 | −0.08 [−0.28, 0.13] | +0.20 [−0.06, 0.47] | **+0.36 [0.11, 0.62] worse** | out of scope | cap 90, window 45, `hi_consistent` |
| FD003 | −0.15 [−0.67, 0.24] | +0.29 [−0.19, 0.72] | **+0.63 [0.17, 1.19] worse** | **+0.21 [0.10, 0.34] worse** | cap 90, window 45, `hi_consistent` |
| FD004 | +0.60 [−0.18, 1.62] | +1.31 [−0.02, 2.78] | **+0.50 [0.13, 0.82] worse** | +0.30 [−0.02, 0.62] | cap 90, window 30, `hi_consistent` + `regime_onehot` |

n = 100 / 260 / 100 / 249 engines.

- **Window:** no longer window beats the A2 winner. The A2 edge flag is resolved: 45 was not
  an artefact of the grid's upper edge. A3 was therefore not re-run.
- **A5a (baseline deviation):** worse on three datasets, not different on FD001.
  - The engine's early-life level adds no usable signal beyond the within-regime z and the
    health index.
  - Its 28 extra columns hurt.
- **A5b (subpopulation probability):**
  - The A4 gate passed on FD003 (AUC 0.997 [0.989, 1.000]; shuffled-label control 0.463
    [0.348, 0.578]) and on FD004 (0.995 [0.989, 0.999]; control 0.475 [0.405, 0.545]).
  - Yet the block is worse on FD003 and not different on FD004.
  - The subpopulation is identifiable early, but knowing it does not improve critical-zone
    accuracy. A plausible reading (not tested) is that the late-life sensors already reveal
    the degradation mode by the time the critical bucket is reached.
- **A4 re-run** (MLflow-logged, with control): it reproduces the Stage A AUCs.
  - 11 of 12 shuffled-label CIs hold 0.5.
  - **Flag:** FD002 at N = 30 gives 0.581 [0.507, 0.660], just above 0.5, while the real AUC
    there is 0.523 [0.445, 0.597], i.e. chance.
  - With 12 single-permutation controls, one 95% CI excluding 0.5 is expected about one time
    in four (1 − 0.975¹² ≈ 0.26). The real AUC is not elevated, so this is not read as
    leakage.
  - No decision depends on it: FD002 is outside the A5b scope, and the FD003/FD004 controls
    hold 0.5.
- **Compute:** 22 configuration runs (43 min of CV run time, 1.8–2.4 min each) + 4 A4 runs,
  serial.

**Carried into Stage B:** the specs above (column "carried spec"). The LSTM runs at cap 90 with
each dataset's window. The baseline channels are not tested in B2, because A5a was adopted
nowhere.

---

## Stage B — LSTM results (registry defaults, cap 90, seed 42, 5 × 3 folds)

Completed 2026-10-04. Full tables are in `reports/selection/stage_B_FD00x.md` / `.json`. Δ is
challenger − incumbent in critical RMSE.

| dataset | B1 adopted `seq_len` (vs 30) | B2 channels (each vs base at the B1 winner) | Stage-B LSTM spec | critical RMSE [95% CI] (n) |
|---|---|---|---|---|
| FD001 | **60** (−0.53 [−0.83, −0.26]; 45: −0.47 [−0.72, −0.22]) | `hi_pooled` +0.04 [−0.24, 0.34] | cap 90, `seq_len` 60, base channels | 3.56 [3.22, 3.93] (100) |
| FD002 | **60** (−0.86 [−1.15, −0.58]; 45: −0.63 [−0.86, −0.39]) | one-hot +0.02 [−0.14, 0.19]; `hi_consistent` **+0.57 [0.37, 0.78] worse** | cap 90, `seq_len` 60, base channels | 3.73 [3.46, 4.01] (260) |
| FD003 | 30 (45: −0.09 [−0.33, 0.17]; 60: **+0.35 [0.02, 0.70] worse**) | `hi_consistent` +0.20 [−0.00, 0.41] | cap 90, `seq_len` 30, base channels | 3.87 [3.48, 4.27] (100) |
| FD004 | 30 (45: −0.11 [−0.33, 0.13]; 60: −0.21 [−0.50, 0.08]) | one-hot −0.12 [−0.43, 0.26]; `hi_consistent` +0.03 [−0.19, 0.25] | cap 90, `seq_len` 30, base channels | 5.74 [5.37, 6.15] (249) |

- **Sequence length:**
  - `seq_len` 20 is worse than 30 everywhere (+1.07 to +1.43).
  - On FD001/FD002 the longest sequence wins. **Flag: 60 is the grid's upper edge on FD001 and
    FD002** (pre-registered: no extension). On FD003, 60 is already worse than 30, so the curve
    turns there.
  - **Decision check for 60 vs 30:** more caught at the 20-cycle budget on FD001 (+4.5
    [0.9, 7.8] pp) and FD002 (+4.9 [1.9, 8.0]); also at 30 cycles on FD002 (+2.6 [0.5, 4.6]).
- **Channels:**
  - No channel block is adopted anywhere. The LSTM learns what the health index and regime
    indicators would give it from the 14 normalized channels.
  - On FD002 the health-index score as a channel is worse.
- **Context, not a ranking.** The ranking is Stage D, at 5 seeds. At one seed, the Stage-B LSTM
  is below the Stage-X XGBoost spec on FD002 (3.73 vs 3.94) and FD004 (5.74 vs 6.11), and above
  it on FD001 (3.56 vs 3.41) and FD003 (3.87 vs 3.45). XGBoost values are its current-code
  re-runs in Stage X.
- **Compute:** 22 LSTM configuration runs, 3.4 h of run time (FD002/FD004 10–13 min each).

**Carried into Stage C:** per dataset, the Stage-X XGBoost spec and the Stage-B LSTM spec above.

---

## Stage C — tuning results (Optuna TPE, seed 42; tuned on repeat 1, evaluated on repeats 2–3)

Completed 2026-10-05. Full tables, including every trial's parameters and objective, are in
`reports/selection/stage_C_FD00x.md` / `.json`.

| dataset | XGBoost: tuned − untuned, critical RMSE (repeats 2–3) | adopted | LSTM: tuned − untuned (repeats 2–3) | adopted |
|---|---|---|---|---|
| FD001 | −0.17 [−0.28, −0.07] | **yes** | −0.72 [−1.10, −0.35] | **yes** |
| FD002 | +0.12 [−0.02, 0.27] | no | +0.03 [−0.19, 0.24] | no |
| FD003 | −0.11 [−0.27, 0.03] | no | −0.31 [−0.62, −0.01] | **yes** |
| FD004 | −0.33 [−0.47, −0.21] | **yes** | +0.15 [−0.17, 0.46] | no |

Each comparison uses 10 splits of the held-out engines (n = 100 / 260 / 100 / 249).

- **Adopted** where the CI excludes 0: XGBoost on FD001/FD004, the LSTM on FD001/FD003.
- **Decision check:** no adopted tuning is worse at any budget.
- **Tuned parameters** (in `params.yaml`-key form in the stage JSONs):
  - The FD001/FD002 LSTM optimum is a larger network: hidden 128, 2 layers, batch 128. The two
    studies replayed the same TPE path.
  - The FD003/FD004 LSTM optima use a single layer.
  - XGBoost optima differ widely by dataset: FD001 depth 10 with strong regularization; FD004
    depth 5, learning rate 0.012.
- **Residual optimism, as pre-registered:**
  - Every repeat-2/3 held-out engine tuned the config in some repeat-1 fold.
  - The non-adoptions on FD002 show the evaluation can still say no.
- **Compute:** 200 trials + 16 full evaluations, 10.0 h of run time.

**Carried into Stage D:** per dataset, the XGBoost and LSTM specs with tuning where adopted.

---

## Stage D — confirmation (5 seeds × 5 × 3 folds) and outcome

Completed 2026-10-05. Full tables are in `reports/selection/stage_D_FD00x.md` / `.json`. The
narrative and figures are in `notebooks/04_model_selection.ipynb`, and the specs in `specs/`.

### Leave-one-out check of every adopted change

Each change is tested as spec − (spec minus the change), on critical RMSE over 75 splits.

| dataset | XGBoost changes kept | XGBoost dropped | LSTM changes kept | LSTM dropped |
|---|---|---|---|---|
| FD001 | cap 90, window 45, `hi_pooled`, tuning | — | cap 90, `seq_len` 60, tuning | — |
| FD002 | cap 90, window 45, `hi_consistent` | — | cap 90, `seq_len` 60 | — |
| FD003 | cap 90, window 45, `hi_consistent` | — | tuning | **cap 90** (cap check) |
| FD004 | cap 90, window 30, `hi_consistent`, tuning | **regime one-hot** (−0.00 [−0.02, 0.01]) | cap 90 | — |

- **Kept changes:** every kept change has a CI entirely below 0. Example:
  - window on FD002: −1.59 [−1.94, −1.25];
  - cap on FD004 XGBoost: −0.63 [−0.78, −0.48];
  - health index on FD003: −0.34 [−0.52, −0.18].
- **FD004 regime one-hot:** dropped by parsimony. It was the weakest screening adoption (Stage A
  CI upper bound −0.01), and Stage D confirms it adds nothing. The FD004 XGBoost finalist is its
  leave-one-out run without it (5 seeds).
- **FD003 LSTM cap — a contradiction (rule 9, X08):**
  - Cap 90 still beats 125 on critical RMSE (−0.24 [−0.34, −0.14]).
  - But at 5 seeds it is **later**: critical-bucket late % +1.94 [0.62, 3.31] pp, urgent-bucket
    late % +1.24 [0.18, 2.34] pp.
  - The pre-registered cap criteria therefore drop it, and the FD003 LSTM finalist trains at cap
    125.
  - The A1-check's cap-90 confirmation (XGBoost, one seed) does not transfer to every model.
  - For XGBoost, the cap criteria hold at 5 seeds on every dataset.

### Finalists and locking

| dataset | XGBoost finalist, critical RMSE [95% CI] | LSTM finalist | Δ LSTM − XGBoost [95% CI] | decision check (Δ caught pp, LSTM − XGBoost; 20 / 30 / 40) | outcome |
|---|---|---|---|---|---|
| FD001 (n = 100) | 3.27 [2.90, 3.62]: cap 90, w45, `hi_pooled`, tuned | 3.25 [2.97, 3.57]: cap 90, L60, tuned | −0.02 [−0.34, 0.28] | −0.2 [−4.1, 3.8]; +1.8 [−0.7, 4.5]; 0.0 | **XGBoost locked** |
| FD002 (n = 260) | 3.94 [3.65, 4.24]: cap 90, w45, `hi_consistent` | 3.94 [3.70, 4.18]: cap 90, L60 | −0.00 [−0.27, 0.27] | −1.3 [−3.7, 0.9]; **−1.2 [−2.3, −0.0]**; −0.1 | **XGBoost locked** |
| FD003 (n = 100) | 3.48 [3.06, 4.01]: cap 90, w45, `hi_consistent` | 3.71 [3.40, 4.06]: cap 125, L30, tuned | +0.22 [−0.33, 0.77] | **−6.4 [−11.5, −1.5]**; **−2.2 [−4.2, −0.3]**; 0.0 | **XGBoost locked** |
| FD004 (n = 249) | 5.73 [5.29, 6.20]: cap 90, w30, `hi_consistent`, tuned | 5.74 [5.37, 6.13]: cap 90, L30 | +0.01 [−0.31, 0.31] | −1.1 [−3.9, 3.3]; −1.3 [−4.7, 0.9]; −0.3 | **XGBoost locked** |

- **Critical RMSE ties on every dataset:** no CI excludes 0, and Wilcoxon p ranges from 0.057 to
  0.85 over 75 splits.
- **The pre-registered tie rule** therefore gives XGBoost, the simpler and cheaper model.
- **The decision check never favours the LSTM.** On FD002 and FD003 XGBoost catches more at
  matched wasted life. So XGBoost is **locked** on all four datasets (`specs/FD00x_xgboost.yaml`,
  `locked: true`). The LSTM specs are `runner-up`.
- **Per subpopulation** (D35), XGBoost / LSTM finalists:
  - FD001: 3.28 / 3.01 (n = 60) and 3.25 / 3.58 (n = 40);
  - FD002: 3.81 / 3.80 (183) and 4.23 / 4.24 (77);
  - FD003: 4.04 / 4.01 (56) and **2.61 [2.33, 2.90] / 3.27 [2.86, 3.75]** (44);
  - FD004: **4.88 [4.44, 5.34] / 5.25 [4.80, 5.68]** (148) and 6.77 [6.05, 7.53] /
    6.38 [5.86, 6.91] (101).

  The two models err on different subpopulations (unpaired CIs, descriptive only). That is a
  candidate for the later decision-layer / ensembling work, not something tested here.
- **D25 ("ship the LSTM") is not supported** on training-engine CV: the LSTM ties and never
  wins. The test-set evidence behind D25 remains inadmissible (rule 3).
- **Compute:** 31 five-seed configuration runs, 13.3 h of run time. Seed-42 splits were reused
  from earlier stages where the spec was identical (D49).

## Stage E — what is delivered, and what is not done

- **Delivered:**
  - `notebooks/04_model_selection.ipynb` (executed; reads only the stage JSONs);
  - this report;
  - `reports/selection/*`;
  - `specs/FD00x_{xgboost,lstm}.yaml`.
- **Not done, by design:**
  - `final_eval` was not run.
  - `params.yaml` defaults are unchanged: still cap 125, window 20, no blocks.
  - `release.spec` / `locked.yaml` were not written.
  - The specs carry `rul_cap: 90`, which `evaluation.spec.load_spec` refuses until
    `params.yaml` `rul_cap` matches. Adopting them is a reviewed, separate step.
  - `train_prod` also refuses feature blocks (`save_bundle`). Serving a health-index spec needs
    the bundle and serving support built first.
- **Open limitations:**
  - One factor at a time (interactions are tested only inside Stage D's leave-one-out).
  - `seq_len` 60 is a grid edge for the FD001/FD002 LSTM runner-ups.
  - Tuning optimism.
  - Screening multiplicity (A–C).
  - The single-permutation A4 control (one of twelve excluded 0.5, see Stage X).

---

## Reproducibility (environment of the Stage A runs)

- Generated: 2026-10-03T19:42:02+00:00
- Git commit: `b290869bfb3d37e613b9d30adc1597da849ec49d`
- DVC data hash (`data/raw`): `43f328008844d7fd56733c63103d09ef.dir`
- Hardware: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 28 logical CPUs; GPU NVIDIA GeForce RTX 4060 Laptop GPU (driver 581.86, CUDA driver 13.0, torch built for CUDA 13.0); resolved devices: torch (LSTM) on cuda, XGBoost on cuda; OS Windows-10-10.0.26200-SP0
- Threads: 20 (torch and all native pools pinned); torch deterministic algorithms: False; CUDA initialized: False
- Python 3.11.15

| Library | Version |
|---|---|
| mlflow | 3.11.1 |
| numpy | 2.4.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.1 |
| threadpoolctl | 3.6.0 |
| torch | 2.11.0+cu130 |
| xgboost | 3.2.0 |

| Native thread pool | Implementation | Version | Threads |
|---|---|---|---|
| libscipy_openblas-64eda39e79589aedb16f58e5547eb599.dll | openblas (blas) | 0.3.30 | 24 |
| libscipy_openblas64_-63c857e738469261263c764a36be9436.dll | openblas (blas) | 0.3.31.188.0 | 24 |
| libiomp5md.dll | openmp (openmp) | None | 20 |
| libiompstubs5md.dll | openmp (openmp) | None | 1 |
| vcomp140.dll | openmp (openmp) | None | 28 |

| Stochastic step | Seed |
|---|---|
| cv_seed | 0 |
| model_seed_42 | 42 |

Pipeline constants in effect: `cfg_REGIME_KMEANS_N_INIT=10`, `cfg_REGIME_KMEANS_SEED=0`, `cfg_SEED=42`, `cfg_SEEDS=(42, 7, 123, 2024, 99)` (full `turbofan.config` in the JSON sidecar).
