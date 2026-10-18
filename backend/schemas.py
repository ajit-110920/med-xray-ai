"""Pydantic schemas for API request validation and response serialization."""

from typing import List, Optional
from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    """Basic root status response."""

    status: str = Field(..., description="API operational status")
    message: str = Field(..., description="Service welcome and descriptive message")
    version: str = Field(..., description="API semantic version")
    model_identifier: str = Field(..., description="Active model checkpoint filename")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="'healthy' if model is loaded and ready, 'unhealthy' otherwise")
    model_loaded: bool = Field(..., description="True if the Keras model is loaded into memory")
    model_identifier: str = Field(..., description="Filename or ID of the loaded model")
    model_path: str = Field(..., description="Filesystem path of the target model checkpoint")
    labels_count: int = Field(..., description="Number of target disease labels loaded")
    input_shape: Optional[List[Optional[int]]] = Field(
        None, description="Expected input tensor shape of the active model"
    )
    error_message: Optional[str] = Field(
        None, description="Error details if model initialization failed"
    )


class LabelsResponse(BaseModel):
    """List of 14 chest disease classes in exact model-output order."""

    labels: List[str] = Field(..., description="Ordered list of disease labels")
    count: int = Field(..., description="Total count of target labels (14)")


class DiseaseScore(BaseModel):
    """Individual disease label paired with raw model activation score."""

    label: str = Field(..., description="Pathology label name")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw output activation score in [0.0, 1.0]. Not calibrated probability.",
    )


class PredictionResponse(BaseModel):
    """Multi-label classification inference response."""

    labels: List[str] = Field(
        ..., description="The 14 disease labels in original model output order"
    )
    scores: List[float] = Field(
        ..., description="The 14 raw activation scores corresponding to labels"
    )
    predictions: List[DiseaseScore] = Field(
        ..., description="List of disease scores in model output order"
    )
    ranked_predictions: List[DiseaseScore] = Field(
        ..., description="List of disease scores sorted from highest to lowest score"
    )
    model_identifier: str = Field(
        ..., description="Model checkpoint identifier used for inference"
    )
    disclaimer: str = Field(
        ..., description="Medical disclaimer stating prototype/educational nature"
    )
