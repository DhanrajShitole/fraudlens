# FraudLens API — production container
FROM python:3.13-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY api/ api/
COPY src/ src/

# Copy the model and metadata required by the API
COPY models/ieee_lightgbm_tuned.joblib models/
COPY models/ieee_lightgbm_tuned_metadata.json models/

# Copy the datasets required for transaction lookup
COPY data/processed/ieee_train.parquet data/processed/
COPY data/processed/ieee_val.parquet data/processed/

# Render supplies PORT; use 8000 locally as the fallback
EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]