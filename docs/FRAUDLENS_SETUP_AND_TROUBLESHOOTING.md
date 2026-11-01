# FraudLens — Complete Setup, Execution & Troubleshooting Guide

**Platform:** Windows 10/11 (PowerShell) / Linux / macOS  
**Target Python Version:** Python 3.13 (Default: `C:\Users\DHANRAJ\AppData\Local\Programs\Python\Python313\python.exe`)  
**Repository Path:** `c:\Users\DHANRAJ\fraudlens`  

---

## 1. System Requirements & Prerequisites

| Resource | Minimum Specification | Recommended Specification |
|---|---|---|
| **Operating System** | Windows 10/11 64-bit, Ubuntu 22.04+, macOS | Windows 11 with PowerShell 7 or WSL2 |
| **Processor** | 4 Cores, x86_64 or ARM64 | 8 Cores (AVX2 support for LightGBM) |
| **System Memory (RAM)** | 8 GB RAM (CPU-only Ollama) | 16 GB RAM (for smooth concurrent LLM inference) |
| **Storage** | 10 GB free SSD disk space | 25 GB free SSD (if retaining full raw Kaggle CSVs) |
| **Python** | Python 3.13.x 64-bit | Python 3.13.9 |
| **Local LLM Runner** | Ollama v0.3+ | Ollama with CUDA / ROCm acceleration |

---

## 2. Step-by-Step Installation & Setup

### Step 1: Clone Repository
```powershell
git clone https://github.com/DhanrajShitole/fraudlens.git
cd fraudlens
```

### Step 2: Install Python Dependencies
Use the exact Python 3.13 executable path:
```powershell
C:\Users\DHANRAJ\AppData\Local\Programs\Python\Python313\python.exe -m pip install --upgrade pip
C:\Users\DHANRAJ\AppData\Local\Programs\Python\Python313\python.exe -m pip install -r requirements.txt pytest reportlab
```

### Step 3: Set Up Ollama for Local Agent Inference
1. Download and install Ollama from [https://ollama.com](https://ollama.com).
2. Start the Ollama background daemon.
3. Pull the required 8-billion parameter model in PowerShell:
   ```powershell
   ollama pull llama3.1:8b
   ```
4. Verify Ollama responds:
   ```powershell
   ollama run llama3.1:8b "Hello"
   ```

---

## 3. Running the Test Suite

Execute the 51 automated unit, integration, drift, and model tests:

```powershell
C:\Users\DHANRAJ\AppData\Local\Programs\Python\Python313\python.exe -m pytest tests/ -v
```

Expected Output:
```text
tests/test_agent_logic.py::... PASSED (19 tests)
tests/test_api.py::... PASSED (12 tests)
tests/test_drift.py::... PASSED (6 tests)
tests/test_model.py::... PASSED (14 tests)
======================= 51 passed, 2 warnings in ~10.7s =======================
```

---

## 4. Running the FraudLens Application

### Step 1: Start the FastAPI Backend
From the repository root (`c:\Users\DHANRAJ\fraudlens`):
```powershell
C:\Users\DHANRAJ\AppData\Local\Programs\Python\Python313\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Verify startup in terminal:
```text
Loading model and metadata...
Loaded model from MLflow registry (fraudlens-ieee-lightgbm@production).
Loading transactions for lookup (train + val combined, matching agent.py's scope)...
Startup complete. model_source=registry:fraudlens-ieee-lightgbm@production
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Step 2: Verify Backend Health
In another terminal:
```powershell
curl http://127.0.0.1:8000/health
```
Expected output:
```json
{"status":"ok","model_loaded":true,"model_type":"LightGBM (tuned, Phase 5 hyperparameters, SMOTE-trained)","model_source":"registry:fraudlens-ieee-lightgbm@production"}
```

### Step 3: Launch the Analyst Dashboard
Double-click `frontend/dashboard.html` or open it in Google Chrome, Microsoft Edge, or Mozilla Firefox:
```powershell
Start-Process "frontend/dashboard.html"
```
1. Verify the green banner shows: `"Connected to FraudLens API."`
2. Enter Transaction ID: `3141064` or `3400379`.
3. Press **Enter** or click **Get Score** to view fraud probability and SHAP reason codes.
4. Click **Investigate** to generate and display the evidence-grounded case file.

---

## 5. Running with Docker Compose (Local Container)

If Docker Desktop is installed on your machine:

```powershell
# Build and run the scoring container
docker compose up --build -d

# Verify container health
curl http://localhost:8000/health

# Stop containers
docker compose down
```

*Note on MLflow in Docker:* `mlflow.db` stores host absolute paths. Inside Docker, the API logs a warning and transparently activates the `joblib-fallback` path, ensuring 100% availability.

---

## 6. Running Offline ML & MLOps Pipelines

### Train Baseline Models (Logistic Regression, LightGBM, Isolation Forest)
```powershell
C:\Users\DHANRAJ\AppData\Local\Programs\Python\Python313\python.exe src/train_baseline.py
```

### Train Tuned Model with SMOTE (Phase 5)
```powershell
C:\Users\DHANRAJ\AppData\Local\Programs\Python\Python313\python.exe src/train_advanced.py
```

### Register Model in MLflow Registry & Assign `@production` Alias
```powershell
C:\Users\DHANRAJ\AppData\Local\Programs\Python\Python313\python.exe src/register_model.py
```

### Execute Feature Drift Detection (PSI Analysis)
```powershell
C:\Users\DHANRAJ\AppData\Local\Programs\Python\Python313\python.exe src/drift_check.py
```

### Run One-Shot Held-Out Test Evaluation (Generates Calibration & Test Reports)
```powershell
C:\Users\DHANRAJ\AppData\Local\Programs\Python\Python313\python.exe src/evaluate_test.py
```

---

## 7. Common Issues & Troubleshooting

### Issue 1: `HTTP 503: Agent investigation failed — the local Ollama service may have crashed or isn't running`
- **Cause:** Ollama background service is not running or the `llama3.1:8b` model has not been downloaded.
- **Resolution:**
  1. Open a new terminal and run: `ollama list`.
  2. If Ollama is stopped, start it from the Windows Start menu or run: `ollama serve`.
  3. Ensure the model is pulled: `ollama pull llama3.1:8b`.
  4. Test that it responds: `ollama run llama3.1:8b "ping"`.

### Issue 2: `Port 8000 already in use`
- **Cause:** A previous Uvicorn process is still running in the background.
- **Resolution:**
  ```powershell
  # Find process using port 8000
  netstat -ano | findstr :8000
  # Terminate process by PID (e.g. 12345)
  taskkill /PID 12345 /F
  ```

### Issue 3: `FileNotFoundError: data/processed/ieee_train.parquet not found`
- **Cause:** Script was executed from inside `src/` rather than the repository root.
- **Resolution:** All FraudLens scripts dynamically resolve paths relative to the repository root. Always run commands from `c:\Users\DHANRAJ\fraudlens`.

### Issue 4: `Warning: Could not load model from MLflow registry... Falling back to local joblib model`
- **Cause:** MLflow database cannot find the original artifact path (normal when moving directories or running inside Docker).
- **Resolution:** This is an intentional self-healing design feature. The API automatically falls back to `models/ieee_lightgbm_tuned.joblib` without any service interruption.

### Issue 5: `SyntaxError: invalid character '—' (U+2014) in source code`
- **Cause:** Source files containing UTF-8 em-dashes were saved with a UTF-8 Byte Order Mark (BOM).
- **Resolution:** All source files in FraudLens are saved as standard UTF-8 without BOM. When editing, ensure your editor saves as UTF-8 (No BOM).

