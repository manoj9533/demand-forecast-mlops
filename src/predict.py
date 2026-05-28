import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from features import build_feature_matrix


def load_model(path: str = "models/forecast_model.joblib"):
    artifact = load(path)
    return artifact["model"], artifact["feature_cols"]


def predict(product_id: str = "PROD_001", days: int = 7) -> dict:
    """Generate a multi-step demand forecast."""
    model, feature_cols = load_model()

    # Simulate recent history for feature engineering
    np.random.seed(hash(product_id) % 1000)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=60, freq="D")
    sales = np.random.normal(120, 15, 60).clip(min=0)
    df = pd.DataFrame({"date": dates, "sales": sales})
    df = build_feature_matrix(df)

    last_row = df[feature_cols].iloc[-1:]
    forecasts = []
    for i in range(days):
        pred = float(model.predict(last_row)[0])
        forecasts.append(round(pred, 2))

    return {"product_id": product_id, "days": days, "forecast": forecasts}


if __name__ == "__main__":
    result = predict()
    print(result)
