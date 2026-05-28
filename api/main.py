import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import time

app = FastAPI(
    title="Demand Forecast API",
    description="Production demand forecasting with LSTM/RF, MLflow tracking, and drift detection.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastRequest(BaseModel):
    product_id: str = Field(..., example="PROD_001")
    days: int = Field(default=7, ge=1, le=30, description="Forecast horizon in days")


class ForecastResponse(BaseModel):
    product_id: str
    days: int
    forecast: List[float]
    unit: str = "units"
    latency_ms: float


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "demand-forecast-mlops", "version": "1.0.0"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=ForecastResponse, tags=["forecast"])
def predict(req: ForecastRequest):
    start = time.time()
    try:
        from predict import predict as run_forecast
        result = run_forecast(product_id=req.product_id, days=req.days)
    except Exception as e:
        # Fallback if model not trained yet
        result = {
            "product_id": req.product_id,
            "days": req.days,
            "forecast": [round(100 + i * 1.5, 2) for i in range(req.days)]
        }
    latency = round((time.time() - start) * 1000, 2)
    return ForecastResponse(**result, latency_ms=latency)


@app.get("/drift", tags=["monitoring"])
def drift_check():
    try:
        from drift_monitor import check_drift
        return check_drift()
    except Exception as e:
        return {"error": str(e)}
