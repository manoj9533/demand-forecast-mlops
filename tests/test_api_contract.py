import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.main import ForecastRequest, health, predict, root


class TestApiContract(unittest.TestCase):
    def test_health_endpoints(self):
        self.assertEqual(root()["status"], "ok")
        self.assertEqual(health()["status"], "healthy")

    def test_predict_shape(self):
        req = ForecastRequest(product_id="PROD_001", days=5)
        resp = predict(req)

        self.assertEqual(resp.product_id, "PROD_001")
        self.assertEqual(resp.days, 5)
        self.assertEqual(len(resp.forecast), 5)
        self.assertGreaterEqual(resp.latency_ms, 0)


if __name__ == "__main__":
    unittest.main()
