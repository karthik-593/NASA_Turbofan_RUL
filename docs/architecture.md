# Architecture

## Package layout

```
src/turbofan/
├── config.py               reads params.yaml (repo root, the single source of every
│                            parameter) strictly at import: RUL_CAP, WINDOW, SEQ_LEN, SEED,
│                            SPLIT_FRAC, SENSORS/KEEP (14), N_REGIMES, FEAT_COLS (42),
│                            model hyperparameters, maintenance buckets
├── data/
│   ├── loader.py            raw C-MAPSS .txt → CMAPSSDataset (train/test/rul_series)
│   └── schema.py            pandera schemas the loader enforces (columns, dtypes, no NaN)
├── features/
│   ├── engineering.py       add_features(): per-regime normalization (k = n_regimes; k=1 = global),
│                            rolling mean/slope → the feature set every model trains on
│   └── envelope.py          training operating envelope; /predict rejects inputs outside it
├── models/
│   ├── baselines.py          MeanBaseline, RidgeRUL, RandomForestRUL
│   ├── xgboost_model.py       XGBoostRUL
│   └── lstm_model.py          LSTMRUL, make_sequences(), make_last_windows()
├── evaluation/
│   ├── metrics.py            rmse, mae, cmapss_score, per_bucket_metrics, late_prediction_pct
│   ├── protocol.py           predict_last_cycle(), score() — the one scoring contract
│   └── comparison.py         build_registry(), run_comparison(), decision_summary(),
│                              run_comparison_multiseed() — the model-selection driver
├── training/
│   ├── train.py               CLI: fit the registry's chosen model, write a bundle
│   └── bundle.py               versioned artifact bundle: save/load/resolve_version
└── serving/
    ├── schemas.py             pydantic request/response models
    ├── service.py              predict_rul(): the serving feature path
    └── app.py                  FastAPI app: GET /health, POST /predict
```

Every model class (`MeanBaseline`, `RidgeRUL`, `RandomForestRUL`, `XGBoostRUL`, `LSTMRUL`)
exposes the same `fit(X_train, y_train, X_val, y_val)` / `predict(X, clip=True)` /
`save(path)` / `load(path)` surface, so `evaluation.comparison` and `training.train` can
treat all five identically through a small registry (`Candidate(factory, kind)`) instead of
branching on model type.

## Data flow

```
raw .txt files
    │  data.loader.load_dataset()
    ▼
train_df, test_df, rul_series          (notebook-style columns: unit, cycle, opN, sN)
    │  evaluation.comparison.split_engines()   — first 80% of engine IDs → train
    ▼
train engines, val engines
    │  features.engineering.add_features()     — fit stats on train, reuse on val/test
    ▼
feat_tr, feat_va, feat_te  (+ stats)
    │  model.fit(...) then predict_last_cycle() / make_last_windows()
    ▼
predictions at each test engine's last cycle
    │  evaluation.protocol.score()              — the one scoring function every path uses
    ▼
{global_rmse, critical_rmse, urgent_rmse, monitor_rmse, healthy_rmse, nasa, late_pct, n}
```

`evaluation.comparison.prepare()` runs the loader → split → `add_features` sequence once
per dataset; `run_comparison()` reuses it across every candidate in the registry so the
split and features are identical for every model in the comparison — only the model itself
varies.

## Artifact bundle

`training.bundle.save_bundle()` writes a self-contained, versioned directory:

```
models/<dataset>/<model>/<version>/
    model.bin           written by the model wrapper's own .save()
    feature_state.pkl   the add_features `stats` (regime clusters + per-regime norm), via joblib
    manifest.json        config snapshot, metrics, seed, lib versions, provenance
```

The feature state ships *with* the model on purpose: a sensor value is meaningless without
knowing which operating regime it was normalized against, so train-time normalization must
be reproduced identically at inference — shipping the model without its `stats` would
silently break every FD002/FD004 prediction. `version` defaults to a UTC timestamp
(lexically sortable, so `max()` finds the latest); `resolve_version(..., "latest")` picks
it up without the caller needing to know the exact string.

## Serving

`serving.app` loads exactly one bundle at process startup (dataset/model/version
configurable by environment variable) and exposes:

- `GET /health` — which bundle is loaded
- `POST /predict` — a chronological cycle history in, `{predicted_rul, maintenance_bucket,
  confidence, n_cycles_used, dataset, model_version}` out

**Train/serve parity is the project's single most important correctness guard.**
`serving.service.predict_rul()` runs the *exact same* `add_features` → windowing pipeline
that training used, fed the bundle's own persisted `stats` — not a re-fit. If these two
paths ever silently drifted (e.g. a re-derived normalization, a different sensor order),
predictions would degrade without any error being raised. `tests/test_serving_parity.py`
guards this directly: it builds the feature window for the same engine through both the
training-path call sequence and the serving-path call sequence (via
`serving.service.to_frame`) and asserts the two arrays are bit-for-bit identical.

## Container & CI

The Docker image (`docker/Dockerfile`) installs CPU-only PyTorch from the PyTorch CPU wheel
index as its own cached layer (no CUDA runtime pulled in, ~700MB smaller than the default
GPU build), then the rest of the pinned runtime dependencies from `requirements.lock`
(deliberately excludes torch — installed separately above), then the package itself. The
trained bundle is baked into the image so the deployed container is self-contained.

CI (`.github/workflows/ci.yml`) runs on every push/PR to `master`: `ruff check` + `ruff
format --check` + `mypy src` (strict) in one job, `pytest` with coverage in a second job
that depends on the first.
