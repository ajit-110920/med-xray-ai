"""Smoke test running inference directly against the actual saved .keras model files.

This verifies that:
1. The actual saved model artifact can be loaded with keras.models.load_model(..., compile=False).
2. Input preprocessing produces the exact expected (1, 224, 224, 3) tensor.
3. Model forward pass executes without crashing on CPU.
4. Output shape is exactly (1, 14) with all values being finite numbers in [0.0, 1.0].
"""

import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from backend.config import settings
from backend.model import ModelManager

BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.mark.smoke
class TestModelArtifactSmoke:
    """Smoke tests using real on-disk .keras model artifacts."""

    @pytest.mark.parametrize("checkpoint_name", ["weighted_best.keras", "finetuned_best.keras"])
    def test_actual_saved_model_loads_and_predicts(self, checkpoint_name: str):
        model_path = BASE_DIR / checkpoint_name
        if not model_path.is_file():
            pytest.skip(f"Checkpoint file {checkpoint_name} not found at {model_path}")

        # Initialize manager for this specific checkpoint
        manager = ModelManager(
            model_path=str(model_path),
            labels_path=str(BASE_DIR / "labels.json"),
        )
        manager.initialize()

        assert manager.is_loaded, f"Model failed to load: {manager.error_message}"
        assert len(manager.labels) == 14, f"Expected 14 labels, got {len(manager.labels)}"
        assert manager.input_shape == (None, 224, 224, 3) or manager.input_shape == [None, 224, 224, 3]
        assert manager.output_shape == (None, 14) or manager.output_shape == [None, 14]

        # Create a synthetic 512x512 grayscale chest X-ray image (typical clinical resolution)
        synthetic_xray = Image.new("L", (512, 512), color=110)
        img_bytes_io = io.BytesIO()
        synthetic_xray.save(img_bytes_io, format="PNG")
        raw_bytes = img_bytes_io.getvalue()

        # Run end-to-end inference
        result = manager.predict(raw_bytes)

        # Assertions on real output
        assert result["model_identifier"] == checkpoint_name
        assert len(result["labels"]) == 14
        assert len(result["scores"]) == 14
        assert len(result["predictions"]) == 14
        assert len(result["ranked_predictions"]) == 14

        # Validate score ranges and finiteness
        for score in result["scores"]:
            assert isinstance(score, float)
            assert np.isfinite(score)
            assert 0.0 <= score <= 1.0

        # Validate that ranked predictions are in non-ascending order
        ranked = result["ranked_predictions"]
        for i in range(len(ranked) - 1):
            assert ranked[i]["score"] >= ranked[i + 1]["score"]
