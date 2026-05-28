import os
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from features import build_feature_matrix
from drift_monitor import compute_psi


class TestFeaturesAndDrift(unittest.TestCase):
    def test_feature_matrix_contains_expected_columns(self):
        dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
        sales = np.linspace(100, 160, 60)
        df = pd.DataFrame({"date": dates, "sales": sales})

        out = build_feature_matrix(df)

        expected_cols = {
            "lag_1",
            "lag_7",
            "rolling_mean_7",
            "rolling_std_7",
            "rolling_mean_30",
            "day_of_week",
            "month",
            "quarter",
            "is_weekend",
        }
        self.assertTrue(expected_cols.issubset(set(out.columns)))
        self.assertGreater(len(out), 0)

    def test_psi_detects_distribution_shift(self):
        np.random.seed(7)
        expected = np.random.normal(100, 10, 1000)
        actual_same = np.random.normal(100, 10, 1000)
        actual_shifted = np.random.normal(130, 10, 1000)

        psi_same = compute_psi(expected, actual_same)
        psi_shifted = compute_psi(expected, actual_shifted)

        self.assertLess(psi_same, psi_shifted)


if __name__ == "__main__":
    unittest.main()
