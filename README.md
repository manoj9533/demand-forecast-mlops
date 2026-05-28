# Demand Forecasting Platform - Practical MLOps Demo

This repository demonstrates an end-to-end demand forecasting workflow with model training, feature engineering, inference API, and drift checks.

## Current Status

- Training pipeline is implemented with `RandomForestRegressor`.
- MLflow logging is supported when an MLflow server is available.
- Drift monitoring is implemented with PSI-based logic (`src/drift_monitor.py`).
- FastAPI endpoints are implemented in `api/main.py`.
- GitHub Actions retraining workflow is present in `.github/workflows/retrain.yml`.

## What This Project Covers

- Time-series feature engineering (lag, rolling, date features)
- Offline training and model artifact persistence
- Online inference through FastAPI (`POST /predict`)
- Drift signal generation for retraining decisions
- Containerized local execution via Docker

## Repository Structure

```text
demand-forecast-mlops/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── retrain.yml
├── api/
│   └── main.py
├── data/
├── models/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── drift_monitor.py
│   ├── features.py
│   ├── predict.py
│   └── train.py
├── tests/
│   ├── test_api_contract.py
│   └── test_features_and_drift.py
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quick Start

```bash
git clone https://github.com/manoj9533/demand-forecast-mlops
cd demand-forecast-mlops
pip install -r requirements.txt
```

## Run Locally

```bash
# Train
python src/train.py --epochs 100 --lookback 30

# API
uvicorn api.main:app --reload

# Drift check
python src/drift_monitor.py
```

## Run Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Docker

```bash
docker build -t forecast-api .
docker run -p 8000:8000 forecast-api
```

## API Endpoints

- `GET /`
- `GET /health`
- `POST /predict`
- `GET /drift`

## Notes for Interview Review

- The current baseline is tree-based forecasting (Random Forest), not LSTM.
- The drift checker currently simulates training vs recent distributions for demonstration.
- The project is structured so real data sources and production registries can be plugged in without changing endpoint contracts.

## Next Improvements

- Add a true sequence model track (LSTM/Temporal CNN) alongside the baseline.
- Add dataset-backed drift reference windows instead of synthetic simulation.
- Add model registry promotion gates in CI/CD.
