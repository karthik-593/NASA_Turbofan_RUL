# Shortcuts over the DVC pipeline (dvc.yaml). `uv run` uses the project environment.
#
#   make data    fetch the DVC-tracked raw C-MAPSS files
#   make repro   dvc repro (needs MLflow: docker compose up -d mlflow)
#   make smoke   tiny isolated end-to-end run, minutes: FD001, 10 engines, 1 split, few epochs
#   make test    the test suite
#   make serve   FastAPI on the latest bundle under models/ (override DATASET= MODEL=)

DATASET ?= FD001
MODEL   ?= lstm
export TURBOFAN_DATASET = $(DATASET)
export TURBOFAN_MODEL   = $(MODEL)

.PHONY: data repro smoke test serve

data:
	uv run dvc pull data/raw.dvc

repro:
	uv run dvc repro

smoke:
	uv run python -m turbofan.pipeline all --smoke

test:
	uv run pytest

serve:
	uv run uvicorn turbofan.serving.app:app
