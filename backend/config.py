"""Configuration module for the Chest X-ray Classification Backend."""

import os
from pathlib import Path
# Standard configuration class without extra dependencies


class Settings:
    """Backend application settings."""

    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # Model and Label paths
    DEFAULT_MODEL_NAME: str = "weighted_best.keras"
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(BASE_DIR / DEFAULT_MODEL_NAME))
    LABELS_PATH: str = os.getenv("LABELS_PATH", str(BASE_DIR / "labels.json"))
    TRAINING_CONFIG_PATH: str = os.getenv(
        "TRAINING_CONFIG_PATH", str(BASE_DIR / "training_config.json")
    )

    # Image Preprocessing Constants
    IMAGE_SIZE: int = 224
    EXPECTED_CHANNELS: int = 3
    EXPECTED_NUM_LABELS: int = 14

    # API Metadata
    API_TITLE: str = "Chest X-ray Multi-Label Disease Classification API"
    API_DESCRIPTION: str = (
        "REST API serving DenseNet121 multi-label chest pathology classification models. "
        "Strictly an educational and research prototype. NOT for clinical or diagnostic use."
    )
    API_VERSION: str = "1.0.0"

    # Server Configuration
    HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("BACKEND_PORT", "8000"))

    # Upload constraints
    MAX_IMAGE_SIZE_MB: int = 15
    ALLOWED_IMAGE_EXTENSIONS: set = {".png", ".jpg", ".jpeg"}
    ALLOWED_MIME_TYPES: set = {"image/png", "image/jpeg", "image/jpg"}

    # Disclaimer text
    DISCLAIMER: str = (
        "RESEARCH AND EDUCATIONAL PROTOTYPE ONLY: The model outputs provided by this system "
        "are raw, uncalibrated activation scores from a research prototype (DenseNet121 trained on NIH ChestX-ray14). "
        "These scores are NOT calibrated diagnostic probabilities and MUST NOT be used for clinical diagnosis, "
        "patient triage, prognosis, or medical treatment planning."
    )


settings = Settings()
