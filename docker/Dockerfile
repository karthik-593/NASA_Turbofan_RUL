# ---- builder ----
FROM python:3.11-slim AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# CPU-only torch as its own layer — heavy (~700 MB uncompressed), rarely changes.
# Version pinned to the working-env torch; installed from the CPU-only wheel index
# so no CUDA runtime is pulled in.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.11.0

# Remaining runtime deps from the pinned lockfile
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# Install the turbofan package itself (--no-deps: lock already covers all deps)
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps .

# ---- runtime ----
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    TURBOFAN_MODELS_DIR=/app/models \
    TURBOFAN_DATASET=FD001 \
    TURBOFAN_MODEL=lstm

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
# Bake the trained bundle into the image so the deployed URL is self-contained.
# Production alternative: mount models/ as a volume and set TURBOFAN_MODELS_DIR.
COPY models/ ./models/

EXPOSE 8000

# $PORT is injected by Cloud Run / Render; falls back to 8000 for local runs.
CMD ["sh", "-c", "uvicorn turbofan.serving.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
