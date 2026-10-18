"""End-to-end integration test simulating frontend-to-backend communication."""

import io
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.model import model_manager

BASE_DIR = Path(__file__).resolve().parent.parent


class TestFrontendBackendIntegration:
    """Verifies that the API serves real model predictions matching frontend expectations."""

    @pytest.fixture(autouse=True)
    def setup_real_model(self):
        """Ensure real model checkpoint is loaded for integration tests."""
        model_path = BASE_DIR / "weighted_best.keras"
        model_manager.model_path = str(model_path)
        model_manager.initialize()
        assert model_manager.is_loaded, f"Model failed to load: {model_manager.error_message}"

    def test_full_pipeline_flow(self):
        with TestClient(app) as client:
            # 1. Frontend checks /health
            health_resp = client.get("/health")
            assert health_resp.status_code == 200
            health_data = health_resp.json()
            assert health_data["status"] == "healthy"
            assert health_data["model_loaded"] is True
            assert health_data["labels_count"] == 14

            # 2. Frontend queries /labels
            labels_resp = client.get("/labels")
            assert labels_resp.status_code == 200
            labels_data = labels_resp.json()
            assert len(labels_data["labels"]) == 14

            # 3. Frontend uploads a chest X-ray and requests /predict
            test_xray = Image.new("L", (512, 512), color=120)
            buf = io.BytesIO()
            test_xray.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            predict_resp = client.post(
                "/predict",
                files={"file": ("integration_scan.png", io.BytesIO(img_bytes), "image/png")},
            )
            assert predict_resp.status_code == 200
            predict_data = predict_resp.json()

            # 4. Verify all payload structures expected by Streamlit dashboard
            assert "ranked_predictions" in predict_data
            assert len(predict_data["ranked_predictions"]) == 14
            assert predict_data["model_identifier"] == "weighted_best.keras"
            assert "disclaimer" in predict_data

            # Top activation must be highest score
            ranked = predict_data["ranked_predictions"]
            assert ranked[0]["score"] >= ranked[1]["score"]
            assert 0.0 <= ranked[0]["score"] <= 1.0

    def test_switching_to_finetuned_checkpoint(self):
        """Verify dynamic switching between checkpoints (weighted_best vs finetuned_best)."""
        finetuned_path = BASE_DIR / "finetuned_best.keras"
        model_manager.load_model(str(finetuned_path))
        assert model_manager.is_loaded
        assert model_manager.model_identifier == "finetuned_best.keras"

        with TestClient(app) as client:
            health_resp = client.get("/health")
            assert health_resp.status_code == 200
            assert health_resp.json()["model_identifier"] == "finetuned_best.keras"

            test_xray = Image.new("RGB", (224, 224), color=(100, 100, 100))
            buf = io.BytesIO()
            test_xray.save(buf, format="JPEG")
            img_bytes = buf.getvalue()

            predict_resp = client.post(
                "/predict",
                files={"file": ("scan.jpg", io.BytesIO(img_bytes), "image/jpeg")},
            )
            assert predict_resp.status_code == 200
            assert predict_resp.json()["model_identifier"] == "finetuned_best.keras"
