import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from joblib import dump
from features import build_feature_matrix

# Optional MLflow (skip silently if tracking server not available)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def generate_sample_data() -> pd.DataFrame:
    """Generate synthetic retail sales data for training."""
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=730, freq="D")
    trend = np.linspace(100, 150, 730)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(730) / 365)
    noise = np.random.normal(0, 5, 730)
    sales = trend + seasonal + noise
    df = pd.DataFrame({"date": dates, "sales": sales.clip(min=0)})
    return df


def train(epochs: int = 100, lookback: int = 30):
    print("=== Demand Forecasting — Training ===")

    # Data
    df = generate_sample_data()
    df = build_feature_matrix(df)

    feature_cols = [c for c in df.columns if c not in ("date", "sales")]
    X = df[feature_cols]
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Model
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"  MAPE : {mape:.2f}%")
    print(f"  RMSE : {rmse:.2f}")

    # Save model
    out = Path("models")
    out.mkdir(exist_ok=True)
    artifact = {"model": model, "feature_cols": feature_cols, "mape": mape, "rmse": rmse}
    dump(artifact, out / "forecast_model.joblib")
    print(f"  Model saved → models/forecast_model.joblib")

    # MLflow logging (optional)
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment("demand-forecasting")
            with mlflow.start_run():
                mlflow.log_param("n_estimators", 200)
                mlflow.log_param("max_depth", 10)
                mlflow.log_param("lookback", lookback)
                mlflow.log_metric("mape", mape)
                mlflow.log_metric("rmse", rmse)
                mlflow.sklearn.log_model(model, "model")
            print("  Metrics logged to MLflow ✓")
        except Exception as e:
            print(f"  MLflow logging skipped: {e}")

    return mape, rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lookback", type=int, default=30)
    args = parser.parse_args()
    train(args.epochs, args.lookback)
