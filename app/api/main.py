from __future__ import annotations

from contextlib import asynccontextmanager
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.service import AICodeDetectionService, FEATURE_MODELS, LANGUAGE_ENCODING, normalize_language


class EvaluateRequest(BaseModel):
    code: str = Field(..., min_length=1, description="Source code to evaluate.")
    language: Optional[str] = Field(None, description="Programming language. Optional if filename is provided.")
    filename: Optional[str] = Field(None, description="Optional filename used for language inference by extension.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    detector_aliases = [
        alias.strip()
        for alias in os.getenv(
            "DETECTOR_MODELS",
            "deepseek-1.3b",
        ).split(",")
        if alias.strip()
    ]

    perturbations = int(os.getenv("DETECTOR_PERTURBATIONS", "5"))
    model_path = os.getenv("MLP_MODEL_PATH")
    threshold_env = os.getenv("MLP_THRESHOLD")
    threshold = float(threshold_env) if threshold_env else None

    app.state.service = await AICodeDetectionService.create(
        detector_aliases=detector_aliases,
        n_perturbations=perturbations,
        model_path=model_path,
        threshold=threshold,
    )
    try:
        yield
    finally:
        if app.state.service is not None:
            await app.state.service.close()
            app.state.service = None


app = FastAPI(
    title="AI Code Detection API",
    version="1.0.0",
    description="ANN-based AI code detection using multi-model feature extraction.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    service = getattr(app.state, "service", None)
    if service is None:
        return {"status": "starting"}

    return {
        "status": "ok",
        "detectors": service.detector_aliases,
        "languages_supported": list(LANGUAGE_ENCODING.keys()),
        "available_detector_models": FEATURE_MODELS,
    }


@app.post("/evaluate")
async def evaluate_code(request: EvaluateRequest) -> dict:
    service = getattr(app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Service is still loading models.")

    if not request.code.strip():
        raise HTTPException(status_code=400, detail="`code` cannot be empty.")

    language = normalize_language(request.language, request.filename)

    try:
        result = await service.evaluate(request.code, language)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
