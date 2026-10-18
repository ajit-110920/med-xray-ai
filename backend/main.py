"""FastAPI application for Chest X-ray Multi-Label Disease Classification."""

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.model import (
    ImageDecodingError,
    InferenceError,
    ModelNotLoadedError,
    model_manager,
)
from backend.schemas import (
    HealthResponse,
    LabelsResponse,
    PredictionResponse,
    StatusResponse,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chest_xray_backend")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan handler to load model once at startup and clean up on shutdown."""
    logger.info("Initializing Chest X-ray classification model backend...")
    model_manager.initialize()
    if model_manager.is_loaded:
        logger.info(
            "Model %s initialized successfully with %d labels.",
            model_manager.model_identifier,
            len(model_manager.labels),
        )
    else:
        logger.warning(
            "Model failed to initialize at startup: %s. /health will report unhealthy.",
            model_manager.error_message,
        )
    yield
    logger.info("Shutting down Chest X-ray classification backend.")


app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
)

# Enable CORS for frontend clients (e.g., Streamlit, browser apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    response_model=StatusResponse,
    summary="Root status",
    tags=["System"],
)
async def root() -> StatusResponse:
    """Return basic API status and welcome message."""
    return StatusResponse(
        status="online",
        message="Chest X-ray Multi-Label Disease Classification API is running.",
        version=settings.API_VERSION,
        model_identifier=model_manager.model_identifier,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
    responses={
        200: {"description": "Service healthy and model loaded"},
        503: {"description": "Service unhealthy or model failed to load"},
    },
)
async def health_check():
    """Return API health status and verify whether the model is loaded in memory.

    Will NOT report healthy if the model failed to load.
    """
    if not model_manager.is_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=HealthResponse(
                status="unhealthy",
                model_loaded=False,
                model_identifier=model_manager.model_identifier,
                model_path=model_manager.model_path,
                labels_count=len(model_manager.labels),
                input_shape=None,
                error_message=model_manager.error_message or "Model is not loaded into memory.",
            ).model_dump(),
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_identifier=model_manager.model_identifier,
        model_path=model_manager.model_path,
        labels_count=len(model_manager.labels),
        input_shape=list(model_manager.input_shape) if model_manager.input_shape else None,
        error_message=None,
    )


@app.get(
    "/labels",
    response_model=LabelsResponse,
    summary="Get disease labels",
    tags=["Model"],
)
async def get_labels() -> LabelsResponse:
    """Return the exact 14 chest pathology labels in their model-output order."""
    if not model_manager.labels:
        # Fallback to load labels directly if not yet in memory
        try:
            model_manager.load_labels()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not load labels: {str(exc)}",
            ) from exc

    return LabelsResponse(
        labels=model_manager.labels,
        count=len(model_manager.labels),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify Chest X-ray image",
    tags=["Inference"],
    responses={
        200: {"description": "Inference successful, returns 14 disease scores"},
        400: {"description": "Invalid, corrupted, or unsupported image file"},
        503: {"description": "Model is not loaded or unavailable"},
        500: {"description": "Inference computation error"},
    },
)
async def predict_xray(
    file: UploadFile = File(..., description="Chest X-ray image (PNG or JPEG)")
) -> PredictionResponse:
    """Accept an uploaded chest X-ray image and return raw model scores for 14 pathologies.

    Preprocessing:
    - Decodes image safely using Pillow
    - Converts color space to RGB
    - Resizes to 224 x 224
    - Normalizes with DenseNet121 ImageNet statistics
    - Runs model forward pass

    Output Note:
    - Raw uncalibrated activation scores in [0.0, 1.0].
    - NOT a medical diagnosis. Prototype / educational research use only.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Classification model is not loaded: {model_manager.error_message or 'Service Unavailable'}",
        )

    # Validate file presence
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file was provided in the upload request.",
        )

    # Basic filename extension check if available
    filename = file.filename.lower()
    allowed_exts = settings.ALLOWED_IMAGE_EXTENSIONS
    if not any(filename.endswith(ext) for ext in allowed_exts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file.filename}'. Allowed formats: {', '.join(sorted(allowed_exts))}",
        )

    # Read uploaded bytes
    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file stream: {str(exc)}",
        ) from exc

    # Enforce file size constraint
    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image file exceeds maximum allowable size of {settings.MAX_IMAGE_SIZE_MB} MB.",
        )

    # Run inference pipeline
    try:
        prediction_result = model_manager.predict(image_bytes)
        return PredictionResponse(**prediction_result)
    except ImageDecodingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(exc)}",
        ) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except InferenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation error: {str(exc)}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during prediction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected internal server error: {str(exc)}",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
    )
