"""
================================================================================
Test Suite: Medical Insurance Cost Category Prediction
Stage 6: Automated End-to-End Verification & Quality Assurance
================================================================================
Covers:
1. Dataset existence, schema, and non-empty checks.
2. Serialized Scikit-Learn pipeline loading and integrity.
3. Model inference on canonical clinical test cases (Low, Medium, High).
4. SQLite persistence layer (insertion, retrieval, analytics summary, reset).
5. Flask REST API routes (/health, /, /predict, /api/history, /api/analytics).
"""

import os
import unittest
import json
import joblib
import pandas as pd

from app import app
import database


class TestMedicalInsuranceProject(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up test environment and Flask test client."""
        cls.client = app.test_client()
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.dataset_path = os.path.join(cls.base_dir, "dataset", "insurance.csv")
        cls.model_path = os.path.join(cls.base_dir, "models", "best_model_pipeline.joblib")
        cls.metadata_path = os.path.join(cls.base_dir, "models", "model_metadata.json")

    # ==========================================================================
    # 1. DATASET TESTS
    # ==========================================================================
    def test_01_dataset_exists_and_valid(self):
        """Verify insurance.csv exists, contains 7 columns, and has > 1300 rows."""
        self.assertTrue(os.path.exists(self.dataset_path), "Dataset file insurance.csv not found!")
        df = pd.read_csv(self.dataset_path)
        self.assertGreaterEqual(len(df), 1300, "Dataset has fewer rows than expected.")
        
        expected_cols = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

        # Check for zero null values
        self.assertEqual(df.isnull().sum().sum(), 0, "Dataset contains unexpected missing values.")

    # ==========================================================================
    # 2. MODEL PIPELINE & ARTIFACT TESTS
    # ==========================================================================
    def test_02_model_bundle_integrity(self):
        """Verify serialized pipeline artifact loads with required keys and classes."""
        self.assertTrue(os.path.exists(self.model_path), "Model pipeline joblib file not found!")
        bundle = joblib.load(self.model_path)
        
        self.assertIn("pipeline", bundle)
        self.assertIn("model_name", bundle)
        self.assertIn("classes", bundle)
        self.assertIn("thresholds", bundle)

        pipeline = bundle["pipeline"]
        classes = list(pipeline.classes_)
        for expected_class in ["High", "Low", "Medium"]:
            self.assertIn(expected_class, classes)

    def test_03_inference_canonical_cases(self):
        """Verify pipeline accurately categorizes clinical edge cases."""
        bundle = joblib.load(self.model_path)
        pipeline = bundle["pipeline"]

        # 1. Young, healthy, non-smoker -> Must be Low Cost
        healthy_patient = pd.DataFrame([{
            "age": 21,
            "sex": "female",
            "bmi": 20.5,
            "children": 0,
            "smoker": "no",
            "region": "northwest"
        }])
        pred_healthy = pipeline.predict(healthy_patient)[0]
        self.assertEqual(pred_healthy, "Low", f"Expected Low cost tier, got {pred_healthy}")

        # 2. Senior smoker with obesity -> Must be High Cost
        highrisk_patient = pd.DataFrame([{
            "age": 55,
            "sex": "male",
            "bmi": 36.5,
            "children": 2,
            "smoker": "yes",
            "region": "southeast"
        }])
        pred_highrisk = pipeline.predict(highrisk_patient)[0]
        self.assertEqual(pred_highrisk, "High", f"Expected High cost tier, got {pred_highrisk}")

    # ==========================================================================
    # 3. SQLITE DATABASE TESTS
    # ==========================================================================
    def test_04_database_crud_operations(self):
        """Verify SQLite insertion, retrieval, and analytics aggregations."""
        # Insert test record
        record_id = database.save_prediction(
            age=33,
            sex="female",
            bmi=25.4,
            children=1,
            smoker="no",
            region="northeast",
            category="Medium",
            confidence=89.5,
            charges_range="$6,250.00 - $12,733.36"
        )
        self.assertIsNotNone(record_id)
        self.assertGreater(record_id, 0)

        # Retrieve recent records
        recent = database.get_recent_predictions(limit=5)
        self.assertTrue(len(recent) > 0)
        self.assertEqual(recent[0]["predicted_category"], "Medium")

        # Check analytics summary
        analytics = database.get_analytics_dashboard_data()
        self.assertIn("kpis", analytics)
        self.assertIn("category_distribution", analytics)
        self.assertIn("smoker_cross", analytics)
        self.assertGreater(analytics["kpis"]["total_predictions"], 0)

    # ==========================================================================
    # 4. FLASK REST API ROUTE TESTS
    # ==========================================================================
    def test_05_health_endpoint(self):
        """Verify GET /api/health returns 200 OK and model_loaded: True."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("model_loaded"))
        self.assertEqual(data.get("status"), "healthy")

    def test_06_homepage_renders(self):
        """Verify GET / renders valid HTML containing project elements."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"MediCost AI", res.data)
        self.assertIn(b"Patient Health Profile", res.data)
        self.assertIn(b"Analytics Dashboard", res.data)

    def test_07_predict_endpoint_valid_json(self):
        """Verify POST /predict returns correct JSON schema for valid patient input."""
        payload = {
            "age": 28,
            "sex": "male",
            "bmi": 23.8,
            "children": 0,
            "smoker": "no",
            "region": "southwest"
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        
        pred = data["prediction"]
        self.assertIn(pred["category"], ["Low", "Medium", "High"])
        self.assertIn("charges_range", pred)
        self.assertIn("confidence", pred)
        self.assertIn("probabilities", pred)
        self.assertIn("insights", pred)

    def test_08_predict_endpoint_input_validation(self):
        """Verify POST /predict rejects invalid demographic boundary values."""
        # 1. Invalid negative age
        res1 = self.client.post("/predict", json={"age": -10, "sex": "male", "bmi": 25, "children": 0, "smoker": "no", "region": "northwest"})
        self.assertEqual(res1.status_code, 400)
        self.assertIn("Age must be between 1 and 120", res1.get_json()["message"])

        # 2. Invalid unrealistic BMI
        res2 = self.client.post("/predict", json={"age": 30, "sex": "male", "bmi": 99.9, "children": 0, "smoker": "no", "region": "northwest"})
        self.assertEqual(res2.status_code, 400)
        self.assertIn("BMI must be a realistic value", res2.get_json()["message"])

        # 3. Invalid smoker string
        res3 = self.client.post("/predict", json={"age": 30, "sex": "male", "bmi": 25, "children": 0, "smoker": "sometimes", "region": "northwest"})
        self.assertEqual(res3.status_code, 400)
        self.assertIn("Smoker status must be 'yes' or 'no'", res3.get_json()["message"])

    def test_09_analytics_api(self):
        """Verify GET /api/analytics returns complete telemetry payload."""
        res = self.client.get("/api/analytics")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["data"]
        self.assertIn("kpis", data)
        self.assertIn("category_distribution", data)
        self.assertIn("bmi_distribution", data)
        self.assertIn("timeline", data)

    def test_10_history_api(self):
        """Verify GET /api/history returns recent audit records."""
        res = self.client.get("/api/history?limit=5")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIsInstance(data["data"], list)


if __name__ == "__main__":
    print("\n" + "#" * 75)
    print(" RUNNING STAGE 6 AUTOMATED END-TO-END TEST SUITE ")
    print("#" * 75 + "\n")
    unittest.main(verbosity=2)
