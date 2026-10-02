# MLflow tracking server. Version pinned to match the client pinned in pyproject.toml
# (mlflow>=2.13.0, currently resolving to 3.11.1 via uv.lock) — bump both together.
FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==3.11.1
EXPOSE 5000
