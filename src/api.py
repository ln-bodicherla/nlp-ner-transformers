"""
FastAPI application for NER inference.

Provides REST endpoints for entity extraction from text,
batch processing, and model information.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    """Request body for single text prediction."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Input text for entity extraction",
    )
    output_format: str = Field(
        default="dict",
        description="Output format: 'dict', 'html', or 'terminal'",
    )


class BatchPredictRequest(BaseModel):
    """Request body for batch prediction."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of input texts for entity extraction",
    )
    batch_size: int = Field(default=32, ge=1, le=128, description="Batch size for inference")


class EntityResponse(BaseModel):
    """Single extracted entity."""
    text: str = Field(description="Entity text span")
    label: str = Field(description="Entity type label")
    start: int = Field(description="Start character offset")
    end: int = Field(description="End character offset")
    confidence: float = Field(description="Prediction confidence score")


class PredictResponse(BaseModel):
    """Response for single text prediction."""
    text: str = Field(description="Original input text")
    entities: List[EntityResponse] = Field(description="Extracted entities")
    entity_count: int = Field(description="Number of entities found")
    labels_found: List[str] = Field(description="Unique entity labels found")
    processing_time_ms: float = Field(description="Processing time in milliseconds")


class BatchPredictResponse(BaseModel):
    """Response for batch prediction."""
    results: List[PredictResponse] = Field(description="Predictions for each input text")
    total_entities: int = Field(description="Total entities across all texts")
    processing_time_ms: float = Field(description="Total processing time in milliseconds")


class LabelsResponse(BaseModel):
    """Response for supported labels endpoint."""
    labels: List[str] = Field(description="Supported entity type labels")
    num_labels: int = Field(description="Number of supported labels")
    label_schema: str = Field(default="BIO", description="Tagging schema used")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether the model is loaded")
    model_path: str = Field(description="Path to the loaded model")
    device: str = Field(description="Inference device")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_path: str
    device: str
    num_labels: int
    supported_labels: List[str]
    model_type: str
    hidden_size: int
    num_attention_heads: int


_inference_engine = None


def get_inference_engine():
    """Return the global inference engine, initializing if needed."""
    global _inference_engine
    if _inference_engine is None:
        model_path = os.environ.get("NER_MODEL_PATH", "./models/ner_model")
        from src.inference import NERInference
        _inference_engine = NERInference(model_path=model_path)
    return _inference_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for model loading."""
    model_path = os.environ.get("NER_MODEL_PATH", "./models/ner_model")
    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        try:
            get_inference_engine()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load model on startup: {e}")
    else:
        logger.warning(f"Model path {model_path} does not exist. Load model before inference.")
    yield
    logger.info("Shutting down NER API")


app = FastAPI(
    title="NER API",
    description="Named Entity Recognition service powered by fine-tuned transformers",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing information."""
    start_time = time.time()
    response = await call_next(request)
    duration = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {duration:.1f}ms"
    )
    return response


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Extract named entities from input text.

    Processes a single text input and returns all detected entities
    with their labels, positions, and confidence scores.
    """
    engine = get_inference_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        entities = engine.predict_text(request.text)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    processing_time = (time.time() - start_time) * 1000

    entity_responses = [EntityResponse(**ent) for ent in entities]
    labels_found = list(set(e.label for e in entity_responses))

    return PredictResponse(
        text=request.text,
        entities=entity_responses,
        entity_count=len(entity_responses),
        labels_found=labels_found,
        processing_time_ms=round(processing_time, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Extract entities from multiple texts in a single request.

    Processes a batch of texts efficiently and returns entity
    predictions for each input.
    """
    engine = get_inference_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        batch_results = engine.predict_batch(
            request.texts, batch_size=request.batch_size
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

    processing_time = (time.time() - start_time) * 1000

    results = []
    total_entities = 0

    for text, entities in zip(request.texts, batch_results):
        entity_responses = [EntityResponse(**ent) for ent in entities]
        labels_found = list(set(e.label for e in entity_responses))
        total_entities += len(entity_responses)

        results.append(PredictResponse(
            text=text,
            entities=entity_responses,
            entity_count=len(entity_responses),
            labels_found=labels_found,
            processing_time_ms=0,
        ))

    return BatchPredictResponse(
        results=results,
        total_entities=total_entities,
        processing_time_ms=round(processing_time, 2),
    )


@app.get("/labels", response_model=LabelsResponse)
async def get_labels() -> LabelsResponse:
    """List all supported entity types.

    Returns the entity labels that the model can predict,
    derived from the model configuration.
    """
    engine = get_inference_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    supported = engine.get_supported_labels()
    return LabelsResponse(
        labels=supported,
        num_labels=len(supported),
        label_schema="BIO",
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health and model readiness.

    Returns the current status of the service including
    whether the model is loaded and operational.
    """
    global _inference_engine

    model_loaded = _inference_engine is not None
    model_path = os.environ.get("NER_MODEL_PATH", "./models/ner_model")
    device = "unknown"

    if _inference_engine is not None:
        device = _inference_engine.device_str

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_path=model_path,
        device=device,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Return detailed model configuration and metadata."""
    engine = get_inference_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = engine.get_model_info()
    return ModelInfoResponse(**info)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions with a structured error response."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )
