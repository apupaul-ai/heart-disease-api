"""FastAPI app that serves heart disease predictions from a saved scikit-learn model."""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.schemas import (FEATURES, HealthResponse, HeartFeatures, InfoResponse,
                         PredictionResponse)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "heart_model.joblib"))
INFO_PATH = MODEL_PATH.with_name("model_info.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once when the server starts (not on every request)
    app.state.model = joblib.load(MODEL_PATH)
    app.state.training_info = json.loads(INFO_PATH.read_text()) if INFO_PATH.exists() else {}
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts the presence of heart disease from medical attributes. "
                "Educational project - not for medical use.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    # Send visitors of the bare URL to the interactive Swagger UI
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Service"])
def health():
    """Liveness check (also used by Render's health check)."""
    return HealthResponse(status="ok", model_loaded=hasattr(app.state, "model"))


@app.get("/info", response_model=InfoResponse, tags=["Service"])
def info():
    """Model type, the list of input features and training details."""
    return InfoResponse(
        model_type=type(app.state.model).__name__,
        n_features=len(FEATURES),
        features=FEATURES,
        target="heart_disease (true = presence, false = absence)",
        training=app.state.training_info,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: HeartFeatures):
    """Predict whether a patient has heart disease."""
    # Build a 1 x 13 array in the same feature order that was used for training
    values = patient.model_dump()
    x = np.array([[values[name] for name in FEATURES]], dtype=float)

    model = app.state.model
    prediction = bool(model.predict(x)[0])
    probability = float(model.predict_proba(x)[0][1])    # probability of class 1 (disease)

    return PredictionResponse(heart_disease=prediction, probability=round(probability, 4))
