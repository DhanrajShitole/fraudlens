FROM python:3.13-slim

WORKDIR /app

# Limit numerical-library thread usage
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV WEB_CONCURRENCY=1
ENV ENABLE_SHAP=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install only API dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Application source
COPY api/ api/
COPY src/ src/

# Model and metadata
COPY models/ieee_lightgbm_tuned.joblib models/
COPY models/ieee_lightgbm_tuned_metadata.json models/

# Generate with scripts/prepare_api_validation_data.py and
# scripts/prepare_investigation_context.py before building.
COPY data/processed/ieee_val_api.parquet data/processed/
COPY data/processed/ieee_investigation_context.parquet data/processed/

EXPOSE 10000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
