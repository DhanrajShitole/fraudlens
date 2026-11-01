# FraudLens — scoring API container
# Base: python:3.13-slim (matches the dev environment)
FROM python:3.13-slim

WORKDIR /app

# Install OS-level build dependencies required by some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first so Docker can cache the layer
COPY requirements.txt .

# Install exact pinned versions from requirements.txt
# (pandas, numpy, scikit-learn, lightgbm, shap, joblib, pyarrow, mlflow,
#  fastapi, uvicorn, ollama — all pinned in requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source — do NOT bake in data/, models/, mlflow.db or mlruns/
# (those are mounted as volumes at runtime via docker-compose.yml)
COPY api/ api/
COPY src/ src/

# Expose the API port
EXPOSE 8000

# Start the FastAPI service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

