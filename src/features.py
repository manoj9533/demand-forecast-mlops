import pandas as pd
import numpy as np


def create_lag_features(df: pd.DataFrame, target_col: str = "sales", lags: int = 7) -> pd.DataFrame:
    """Create lag features for time series."""
    df = df.copy()
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str = "sales") -> pd.DataFrame:
    """Create rolling mean/std features."""
    df = df.copy()
    df["rolling_mean_7"] = df[target_col].shift(1).rolling(7).mean()
    df["rolling_std_7"] = df[target_col].shift(1).rolling(7).std()
    df["rolling_mean_30"] = df[target_col].shift(1).rolling(30).mean()
    return df


def add_date_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Extract date-based features."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = add_date_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df.dropna(inplace=True)
    return df
