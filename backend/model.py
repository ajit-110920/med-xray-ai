"""Model loading, preprocessing, and inference management."""

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
import keras
from keras.applications.densenet import preprocess_input

from backend.config import settings

logger = logging.getLogger("chest_xray_backend.model")


class ModelError(Exception):
    """Base exception for model-related errors."""


class ImageDecodingError(ModelError):
    """Raised when an uploaded file cannot be parsed as a valid image."""


class ModelNotLoadedError(ModelError):
    """Raised when inference is attempted before a model is loaded."""


class InferenceError(ModelError):
    """Raised when model inference fails or outputs invalid shapes/values."""


class ModelManager:
    """Manages model lifecycle, input preprocessing, and multi-label prediction."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        labels_path: Optional[str] = None,
    ) -> None:
        self.model_path = model_path or settings.MODEL_PATH
        self.labels_path = labels_path or settings.LABELS_PATH
        self.model: Optional[keras.Model] = None
        self.labels: List[str] = []
        self.input_shape: Optional[Tuple[Any, ...]] = None
        self.output_shape: Optional[Tuple[Any, ...]] = None
        self.error_message: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        """Return True if model and labels are successfully loaded into memory."""
        return self.model is not None and len(self.labels) > 0

    @property
    def model_identifier(self) -> str:
        """Return the filename of the loaded model checkpoint."""
        return Path(self.model_path).name

    def load_labels(self) -> List[str]:
        """Load and validate the 14 disease labels from JSON."""
        path = Path(self.labels_path)
        if not path.is_file():
            raise FileNotFoundError(f"Labels file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            labels_data = json.load(f)

        if not isinstance(labels_data, list):
            raise ValueError(f"Expected labels JSON to be a list, got {type(labels_data)}")

        if len(labels_data) != settings.EXPECTED_NUM_LABELS:
            logger.warning(
                "Labels count (%d) differs from expected (%d)",
                len(labels_data),
                settings.EXPECTED_NUM_LABELS,
            )

        self.labels = [str(lbl) for lbl in labels_data]
        logger.info("Loaded %d labels from %s", len(self.labels), path)
        return self.labels

    def load_model(self, model_path: Optional[str] = None) -> keras.Model:
        """Load the Keras model checkpoint into memory.

        Uses compile=False so custom loss functions (e.g. weighted_binary_crossentropy)
        used during training do not break inference deserialization.
        """
        if model_path:
            self.model_path = model_path

        path = Path(self.model_path)
        if not path.is_file():
            self.error_message = f"Model checkpoint file not found at: {path}"
            logger.error(self.error_message)
            raise FileNotFoundError(self.error_message)

        try:
            logger.info("Loading Keras model from %s...", path)
            # compile=False allows loading weighted_best.keras without defining custom loss
            loaded_model = keras.models.load_model(str(path), compile=False)
            self.model = loaded_model
            self.input_shape = getattr(loaded_model, "input_shape", None)
            self.output_shape = getattr(loaded_model, "output_shape", None)
            self.error_message = None

            logger.info(
                "Model loaded successfully. Input shape: %s, Output shape: %s",
                self.input_shape,
                self.output_shape,
            )
            return loaded_model
        except Exception as exc:
            self.model = None
            self.error_message = f"Failed to load model from {path}: {str(exc)}"
            logger.exception(self.error_message)
            raise ModelError(self.error_message) from exc

    def initialize(self) -> None:
        """Load both labels and model checkpoint. Records errors instead of crashing startup."""
        try:
            self.load_labels()
        except Exception as exc:
            self.error_message = f"Error loading labels: {exc}"
            logger.error(self.error_message)

        try:
            self.load_model()
        except Exception as exc:
            # error_message is recorded in load_model
            logger.error("Model initialization deferred due to load failure: %s", exc)

    def preprocess_image(
        self, image_bytes: bytes, target_size: Tuple[int, int] = (settings.IMAGE_SIZE, settings.IMAGE_SIZE)
    ) -> np.ndarray:
        """Safely decode, convert, resize, and apply DenseNet121 preprocessing.

        Steps:
        1. Decode raw bytes with Pillow and catch invalid/corrupt image data.
        2. Convert image to 3-channel RGB.
        3. Resize using bilinear interpolation to (224, 224).
        4. Convert to float32 NumPy array.
        5. Apply keras.applications.densenet.preprocess_input.
        6. Expand batch dimension to (1, 224, 224, 3).
        """
        if not image_bytes or len(image_bytes) == 0:
            raise ImageDecodingError("Uploaded image file is empty (0 bytes).")

        try:
            # Safe image parsing with Pillow
            with Image.open(io.BytesIO(image_bytes)) as img:
                # Force loading raster data into memory to verify image is not truncated
                img.load()

                # Supported format check
                img_format = (img.format or "").upper()
                if img_format not in {"PNG", "JPEG", "JPG"}:
                    logger.warning("Uploaded image format is %s, proceeding with RGB conversion", img_format)

                # Ensure 3-channel RGB (NIH chest X-rays are often grayscale 'L' or 'RGBA')
                rgb_img = img.convert("RGB")

                # Resize to target resolution (224, 224)
                resized_img = rgb_img.resize(target_size, Image.Resampling.BILINEAR)

                # Convert to floating-point NumPy array
                np_img = np.array(resized_img, dtype=np.float32)

        except (UnidentifiedImageError, OSError, ValueError) as exc:
            logger.warning("Image decoding failed: %s", exc)
            raise ImageDecodingError(f"Could not decode image file: {str(exc)}") from exc
        except Exception as exc:
            logger.error("Unexpected error reading image: %s", exc)
            raise ImageDecodingError(f"Invalid image content: {str(exc)}") from exc

        # Apply DenseNet121 standard ImageNet normalization
        # Scales to [0, 1] and normalizes with ImageNet mean and std
        preprocessed = preprocess_input(np_img)

        # Add batch dimension: (1, 224, 224, 3)
        batch = np.expand_dims(preprocessed, axis=0)
        return batch

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Run end-to-end inference on raw image bytes and return disease scores."""
        if not self.is_loaded or self.model is None:
            raise ModelNotLoadedError(
                f"Model is not loaded. Current status: {self.error_message or 'Unloaded'}"
            )

        # 1. Preprocess input
        img_batch = self.preprocess_image(image_bytes)

        # 2. Run model forward pass
        try:
            # Using model(x, training=False) is faster than model.predict for single batches
            preds_tensor = self.model(img_batch, training=False)
            if hasattr(preds_tensor, "numpy"):
                raw_preds = preds_tensor.numpy()
            else:
                raw_preds = np.array(preds_tensor)
        except Exception as exc:
            logger.exception("Model forward pass failed: %s", exc)
            raise InferenceError(f"Inference execution failed: {str(exc)}") from exc

        # 3. Validate output shape and values
        expected_labels_count = len(self.labels)
        if raw_preds.ndim != 2 or raw_preds.shape[1] != expected_labels_count:
            raise InferenceError(
                f"Expected output shape (1, {expected_labels_count}), got {raw_preds.shape}"
            )

        scores_1d = raw_preds[0]
        if np.isnan(scores_1d).any() or np.isinf(scores_1d).any():
            raise InferenceError("Model output contains NaN or Inf values.")

        # Convert to native floats
        scores_list = [float(s) for s in scores_1d]

        # Structure ordered and ranked predictions
        predictions = [
            {"label": lbl, "score": score}
            for lbl, score in zip(self.labels, scores_list)
        ]
        ranked_predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)

        return {
            "labels": self.labels,
            "scores": scores_list,
            "predictions": predictions,
            "ranked_predictions": ranked_predictions,
            "model_identifier": self.model_identifier,
            "disclaimer": settings.DISCLAIMER,
        }


# Global model manager instance
model_manager = ModelManager()
