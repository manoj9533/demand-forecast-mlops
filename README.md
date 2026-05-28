# Demand Forecasting Platform — End-to-End MLOps

A production ML system I built to replace manual Excel-based demand projections with an automated LSTM forecasting pipeline — complete with experiment tracking, drift monitoring, CI/CD retraining, and a live inference API.

---

## Why I Built This

The business problem was simple: 3 business units were still doing demand forecasting manually in Excel. Projections were slow, inconsistent, and had no feedback loop. I wanted to build something that could run reliably in production with minimal human intervention.

---

## What It Does

- Trains an LSTM model on historical retail sales data (18 months, multi-SKU)
- Tracks all experiments in MLflow — model architecture, features, hyperparameters, metrics
- Monitors production data for drift using Evidently AI (PSI-based)
- Automatically retrains via GitHub Actions when drift is detected
- Serves predictions through a FastAPI REST endpoint on AWS EC2
- Fully containerised with Docker

---

## Architecture

```
S3 (Raw Data)
    │
    ▼
ETL + Feature Engineering
    │
    ▼
LSTM Training  →  MLflow (experiments, metrics, artifacts)
    │
    ▼
Model Registry  →  Staging → Production promotion
    │
    ▼
Evidently AI (PSI drift monitor)
    │  PSI > 0.2
    ▼
GitHub Actions  →  Auto retrain + redeploy
    │
    ▼
FastAPI on AWS EC2  →  /predict endpoint
```

---

## Tech Stack

- **Model:** TensorFlow / Keras (LSTM), benchmarked against ARIMA and XGBoost
- **Tracking:** MLflow
- **Drift Detection:** Evidently AI
- **API:** FastAPI
- **Infra:** Docker, GitHub Actions, AWS EC2, AWS S3

---

## Model Results

| Model | MAPE | Notes |
|---|---|---|
| LSTM | 8.2% | Final production model |
| XGBoost | 11.4% | Strong baseline |
| ARIMA | 15.7% | Struggled with non-linear patterns |

Ran 30+ MLflow experiments across lookback windows, layer sizes, dropout rates, and feature sets before landing on the final architecture.

---

## MLOps Pipeline

**Experiment Tracking**
Every training run is logged to MLflow — loss curves, MAPE, RMSE, artifacts. Model versions are promoted from staging to production through the MLflow registry after evaluation passes.

**Drift Detection**
Evidently AI compares incoming inference data against the training distribution weekly. I used PSI (Population Stability Index) with a threshold of 0.2. When it crosses that, a GitHub Actions workflow kicks off automatically — no manual trigger needed. This brought the model degradation window down from weeks to under 24 hours.

**CI/CD Retraining**
The GitHub Actions workflow pulls fresh data from S3, retrains the model, logs it to MLflow, runs evaluation, and redeploys the Docker container on EC2 if metrics pass.

**Inference API**
FastAPI endpoint (`POST /predict`) takes a product ID and date range, returns multi-step forecasts with confidence intervals. Handles 500+ daily requests at under 100ms p95 latency.

---

## Repo Structure

```
demand-forecast-mlops/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_lstm_training.ipynb
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── drift_monitor.py
│   └── features.py
├── api/
│   └── main.py
├── .github/
│   └── workflows/
│       └── retrain.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Running Locally

```bash
git clone https://github.com/manojkumargowd/demand-forecast-mlops
cd demand-forecast-mlops
pip install -r requirements.txt

# Train
python src/train.py --epochs 50 --lookback 30

# MLflow UI
mlflow ui --port 5000

# API
uvicorn api.main:app --reload

# Drift check
python src/drift_monitor.py
```

**Docker**
```bash
docker build -t forecast-api .
docker run -p 8000:8000 forecast-api
```

---

## Results

- 92% MAPE accuracy on 18-month retail dataset
- Model degradation window: weeks → 24 hours
- 30+ experiments tracked end-to-end in MLflow
- API p95 latency under 100ms
- Deployment cycle cut from 2 weeks to 3 days

---

## What I'd Improve Next

- Add Prophet for SKUs with strong seasonality
- Anomaly detection for sudden demand spikes (promo events, etc.)
- Grafana dashboard for live monitoring visibility
- Multi-tenant support across business units


