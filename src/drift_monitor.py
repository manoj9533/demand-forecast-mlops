import numpy as np
import pandas as pd
from pathlib import Path

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index — measures distribution shift."""
    def scale(x, buckets):
        breakpoints = np.linspace(min(x.min(), actual.min()), max(x.max(), actual.max()), buckets + 1)
        counts = np.histogram(x, breakpoints)[0]
        return counts / len(x) + 1e-9

    expected_perc = scale(expected, buckets)
    actual_perc = scale(actual, buckets)
    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return round(float(psi), 4)


def check_drift(threshold: float = 0.2) -> dict:
    """Compare recent inference data against training distribution."""
    np.random.seed(42)
    training_data = np.random.normal(100, 15, 500)
    # Simulate drift: mean has shifted
    recent_data = np.random.normal(120, 18, 200)

    psi = compute_psi(training_data, recent_data)
    drift_detected = psi > threshold

    result = {
        "psi_score": psi,
        "threshold": threshold,
        "drift_detected": drift_detected,
        "action": "retrain" if drift_detected else "ok"
    }

    print(f"PSI Score    : {psi}")
    print(f"Drift Detected: {drift_detected}")
    print(f"Action        : {result['action']}")
    return result


if __name__ == "__main__":
    check_drift()
