# Turbofan RUL Prediction

Predictive maintenance on the NASA C-MAPSS turbofan dataset: predict how many operating
cycles an aircraft engine has left before failure, from its sensor history, and serve that
prediction as a maintenance decision — a bucket (critical / urgent / monitor / healthy) and
a confidence band, not just a bare number.

Full write-up of the reasoning behind each decision below lives in [`docs/`](docs/):
[problem framing](docs/problem_framing.md), [architecture](docs/architecture.md),
[results](docs/results.md).

## Result

Five candidates (mean baseline, ridge, random forest, XGBoost, LSTM) were run through one
evaluation protocol across all four C-MAPSS sub-datasets, then the top two were re-run over
5 seeds to check the win survives model-init variance. **LSTM ships to production**: pooled
critical-zone RMSE 4.65 vs XGBoost's 6.92, winning untuned against a tuned XGBoost, with the
lead surviving ±1σ on 3 of 4 datasets. Full evidence and the explicit list of what this
comparison does *not* cover: [`docs/results.md`](docs/results.md).

| Dataset | Best model | Critical-zone RMSE |
|---|---|---:|
| FD001 | LSTM | 3.13 |
| FD002 | XGBoost | 5.64 |
| FD003 | LSTM | 4.27 |
| FD004 | LSTM | 6.48 |

## Why these design choices

I spent years as a Senior Assistant Loco Pilot on Indian Railways, operating and
troubleshooting electric locomotives. Three decisions in this project came directly out of
that experience, not out of a textbook:

**Critical-zone RMSE as the headline metric, not global RMSE.** A prediction error on an
engine with 200 cycles left and one with 10 cycles left are not the same mistake. On a real
locomotive, nobody cares if your read on a healthy loco is off by a few units — you care a
great deal when a fault reading is close to the point where you'd actually pull it from
service. Optimizing a single pooled RMSE treats those two errors as equally important, which
is exactly backwards from how a maintenance decision actually works. This project scores
models on critical-zone [0–25] RMSE specifically, with the standard whole-range RMSE
reported alongside but not optimized for.

**Per-regime normalization as a correctness requirement, not a modeling nicety.** A gauge
reading on a locomotive only means something relative to what notch/throttle setting it was
taken under — the same current draw is normal at full power and alarming standing still. The
FD002/FD004 sub-datasets have the identical structure: six operating regimes, each with its
own sensor baseline. Normalize globally and you're comparing readings across incompatible
baselines, which turns a real degradation trend into what looks like noise. `add_features`
fits a 6-cluster KMeans on the operating settings and normalizes each sensor within its own
regime — recovering that structure mattered more to the final result than any amount of
model tuning downstream.

**Maintenance buckets, not a raw cycle count.** Nobody scheduling maintenance on rolling
stock works from a bare regression output. They work from an actionable category: does this
need attention now, can it wait for the next scheduled window, or is it fine. The serving
API's `/predict` endpoint returns `maintenance_bucket` and a `confidence` band (the shipped
model's own measured error in that specific bucket) alongside the raw number, because the
number alone isn't the deliverable — the decision is.

## Project layout

```
src/turbofan/
├── config.py          single source of truth for RUL cap, sensor list, feature columns, seeds
├── data/loader.py      raw C-MAPSS .txt → structured train/test/rul dataframes
├── features/            per-regime/global normalization + rolling mean/slope
├── models/               MeanBaseline, RidgeRUL, RandomForestRUL, XGBoostRUL, LSTMRUL
├── evaluation/            metrics, scoring protocol, multi-seed model comparison
├── training/               CLI to fit a model and write a versioned artifact bundle
└── serving/                 FastAPI app: GET /health, POST /predict
```

See [`docs/architecture.md`](docs/architecture.md) for the full data flow, the artifact
bundle format, and the train/serve parity guarantee.

## Quickstart

```bash
uv sync --all-extras                # install deps
uv run dvc pull                     # fetch data/raw (NASA C-MAPSS files, DVC-tracked)

# train and ship the LSTM for one dataset
uv run python -m turbofan.training.train --dataset FD001

# serve it
TURBOFAN_DATASET=FD001 TURBOFAN_MODEL=lstm uv run uvicorn turbofan.serving.app:app

# or containerized (CPU-only torch, no CUDA runtime)
docker build -f docker/Dockerfile -t turbofan-rul .
docker run -p 8000:8000 turbofan-rul
```

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"cycles": [...]}'   # 30+ cycles of {op1, op2, op3, sensors: {...}}
```

## Development

```bash
uv sync --all-extras
uv run pytest tests/ -v --cov=src/turbofan
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
```

### Data

`data/raw` (the NASA C-MAPSS files) is tracked with DVC, not git — `data/raw.dvc` is the
committed pointer. The default remote is a local directory (`../dvc-store`, sibling of the
repo), swappable to a cloud remote later without changing how data is tracked
(`docs/decisions.md`). Run `dvc pull` to fetch it; `dvc push` after adding or changing data.

Tests marked `requires_data` (`tests/test_data_integrity.py`, `tests/test_serving_parity.py`)
need `data/raw`. If it's absent they skip with reason "data not pulled" — except when
`REQUIRE_DATA_TESTS=1` is set, which turns that into a hard failure instead of a silent skip.
CI (GitHub Actions) leaves it unset, since it doesn't pull the data; set it in environments
(e.g. Jenkins) where the data is expected to be present. `tests/test_api.py` needs neither
`data/raw` nor `models/` — it trains a tiny LSTM on synthetic data to test the API contract;
real-bundle behaviour is covered by the container smoke test.

CI (`.github/workflows/ci.yml`) runs all four on every push/PR to `master`.
