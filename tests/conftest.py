"""Test fixtures and mock helpers for the test suite."""

import io
from pathlib import Path
from typing import Generator
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app
from backend.model import model_manager

BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """TestClient instance for API endpoints with mocked model state."""
    # Ensure model_manager has labels loaded
    if not model_manager.labels:
        model_manager.load_labels()
    
    with TestClient(app) as client:
        yield client


@pytest.fixture
def valid_png_bytes() -> bytes:
    """Generate a valid 256x256 RGB PNG image in memory."""
    img = Image.new("RGB", (256, 256), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def valid_grayscale_png_bytes() -> bytes:
    """Generate a valid 256x256 grayscale (mode L) PNG image in memory."""
    img = Image.new("L", (256, 256), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    """Generate a valid 300x300 JPEG image in memory."""
    img = Image.new("RGB", (300, 300), color=(180, 150, 130))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def corrupt_image_bytes() -> bytes:
    """Generate corrupt bytes pretending to be an image."""
    return b"NotAnImageHeader" + b"\x00\xFF" * 50


@pytest.fixture
def mock_inference_scores() -> np.ndarray:
    """Deterministic mock inference output vector for 14 classes."""
    return np.array(
        [
            [
                0.28,  # Atelectasis
                0.15,  # Cardiomegaly
                0.62,  # Consolidation
                0.35,  # Edema
                0.40,  # Effusion
                0.03,  # Emphysema
                0.20,  # Fibrosis
                0.05,  # Hernia
                0.45,  # Infiltration
                0.12,  # Mass
                0.30,  # Nodule
                0.09,  # Pleural_Thickening
                0.18,  # Pneumonia
                0.25,  # Pneumothorax
            ]
        ],
        dtype=np.float32,
    )
