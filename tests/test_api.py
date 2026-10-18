"""Unit and integration tests for FastAPI backend endpoints."""

import io
from unittest.mock import MagicMock
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.model import model_manager


class TestRootEndpoint:
    """Tests for GET / endpoint."""

    def test_root_returns_status_online(self, test_client: TestClient):
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert "message" in data
        assert "version" in data
        assert "model_identifier" in data


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_reports_healthy_when_model_loaded(self, test_client: TestClient, monkeypatch):
        # Ensure model manager is seen as loaded
        monkeypatch.setattr(model_manager, "model", MagicMock())
        monkeypatch.setattr(model_manager, "labels", ["L" + str(i) for i in range(14)])
        monkeypatch.setattr(model_manager, "input_shape", (None, 224, 224, 3))
        monkeypatch.setattr(model_manager, "error_message", None)

        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["labels_count"] == 14
        assert data["error_message"] is None

    def test_health_reports_unhealthy_when_model_not_loaded(self, test_client: TestClient, monkeypatch):
        # Simulate model load failure
        monkeypatch.setattr(model_manager, "model", None)
        monkeypatch.setattr(model_manager, "error_message", "Failed to load model checkpoint")

        response = test_client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False
        assert "Failed to load" in (data["error_message"] or "")


class TestLabelsEndpoint:
    """Tests for GET /labels endpoint."""

    def test_labels_returns_14_exact_labels(self, test_client: TestClient):
        response = test_client.get("/labels")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 14
        assert len(data["labels"]) == 14

        # Validate against known NIH ChestX-ray14 labels
        expected_first = "Atelectasis"
        expected_last = "Pneumothorax"
        assert data["labels"][0] == expected_first
        assert data["labels"][-1] == expected_last
        assert "Cardiomegaly" in data["labels"]
        assert "Pneumonia" in data["labels"]
        assert "Pleural_Thickening" in data["labels"]


class TestPredictEndpoint:
    """Tests for POST /predict endpoint with mocked model forward pass."""

    def test_predict_valid_png_success(
        self, test_client: TestClient, valid_png_bytes: bytes, mock_inference_scores: np.ndarray, monkeypatch
    ):
        # Mock model call to return deterministic output without invoking heavy TF
        mock_model = MagicMock(return_value=mock_inference_scores)
        monkeypatch.setattr(model_manager, "model", mock_model)
        monkeypatch.setattr(model_manager, "labels", [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
            "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
            "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
        ])

        files = {"file": ("xray.png", io.BytesIO(valid_png_bytes), "image/png")}
        response = test_client.post("/predict", files=files)

        assert response.status_code == 200
        data = response.json()

        # Validate schema fields
        assert "labels" in data
        assert "scores" in data
        assert "predictions" in data
        assert "ranked_predictions" in data
        assert "model_identifier" in data
        assert "disclaimer" in data

        # Check label count and alignment
        assert len(data["labels"]) == 14
        assert len(data["scores"]) == 14
        assert len(data["predictions"]) == 14
        assert len(data["ranked_predictions"]) == 14

        # Check score values are finite and within [0, 1]
        for score in data["scores"]:
            assert 0.0 <= score <= 1.0

        # Check ranking order (highest to lowest)
        ranked = data["ranked_predictions"]
        for i in range(len(ranked) - 1):
            assert ranked[i]["score"] >= ranked[i + 1]["score"]

        # Check disclaimer is present and explicit
        assert "NOT" in data["disclaimer"] or "not" in data["disclaimer"]
        assert "diagnosis" in data["disclaimer"].lower()

    def test_predict_grayscale_image_success(
        self, test_client: TestClient, valid_grayscale_png_bytes: bytes, mock_inference_scores: np.ndarray, monkeypatch
    ):
        """Grayscale chest X-rays should be safely converted to 3-channel RGB."""
        mock_model = MagicMock(return_value=mock_inference_scores)
        monkeypatch.setattr(model_manager, "model", mock_model)

        files = {"file": ("chest_gray.png", io.BytesIO(valid_grayscale_png_bytes), "image/png")}
        response = test_client.post("/predict", files=files)
        assert response.status_code == 200
        data = response.json()
        assert len(data["scores"]) == 14

    def test_predict_jpeg_image_success(
        self, test_client: TestClient, valid_jpeg_bytes: bytes, mock_inference_scores: np.ndarray, monkeypatch
    ):
        mock_model = MagicMock(return_value=mock_inference_scores)
        monkeypatch.setattr(model_manager, "model", mock_model)

        files = {"file": ("patient_scan.jpg", io.BytesIO(valid_jpeg_bytes), "image/jpeg")}
        response = test_client.post("/predict", files=files)
        assert response.status_code == 200

    def test_predict_rejects_unsupported_extension(self, test_client: TestClient):
        files = {"file": ("report.txt", io.BytesIO(b"Patient Report Text"), "text/plain")}
        response = test_client.post("/predict", files=files)
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]

    def test_predict_rejects_empty_file(self, test_client: TestClient):
        files = {"file": ("empty.png", io.BytesIO(b""), "image/png")}
        response = test_client.post("/predict", files=files)
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_predict_rejects_corrupted_image(self, test_client: TestClient, corrupt_image_bytes: bytes):
        files = {"file": ("broken.png", io.BytesIO(corrupt_image_bytes), "image/png")}
        response = test_client.post("/predict", files=files)
        assert response.status_code == 400
        assert "Invalid image" in response.json()["detail"] or "decode" in response.json()["detail"].lower()

    def test_predict_returns_503_when_model_unloaded(
        self, test_client: TestClient, valid_png_bytes: bytes, monkeypatch
    ):
        monkeypatch.setattr(model_manager, "model", None)
        monkeypatch.setattr(model_manager, "error_message", "Checkpoint unavailable")

        files = {"file": ("xray.png", io.BytesIO(valid_png_bytes), "image/png")}
        response = test_client.post("/predict", files=files)
        assert response.status_code == 503
        assert "not loaded" in response.json()["detail"].lower()

    def test_predict_handles_nan_or_inf_in_model_output(
        self, test_client: TestClient, valid_png_bytes: bytes, monkeypatch
    ):
        # Simulate faulty model output producing NaN
        nan_output = np.array([[np.nan] * 14], dtype=np.float32)
        mock_model = MagicMock(return_value=nan_output)
        monkeypatch.setattr(model_manager, "model", mock_model)

        files = {"file": ("xray.png", io.BytesIO(valid_png_bytes), "image/png")}
        response = test_client.post("/predict", files=files)
        assert response.status_code == 500
        assert "NaN or Inf" in response.json()["detail"]
