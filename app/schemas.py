"""Pydantic models describing the API's input and output."""
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field

# Order in which the features are fed to the model (must match train_model.py)
FEATURES: List[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


class HeartFeatures(BaseModel):
    """Medical attributes of one patient."""

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex: 1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., gt=0, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., gt=0, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl: 1 = yes, 0 = no")
    restecg: int = Field(..., ge=0, le=2, description="Resting electrocardiographic results (0-2)")
    thalach: float = Field(..., gt=0, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina: 1 = yes, 0 = no")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise relative to rest")
    slope: int = Field(..., ge=0, le=2, description="Slope of the peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-4) colored by fluoroscopy")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia: 0 = missing/unknown, 1 = fixed defect, 2 = normal, 3 = reversible defect")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
                    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
                }
            ]
        }
    )


class PredictionResponse(BaseModel):
    heart_disease: bool = Field(..., description="True = heart disease predicted, False = not predicted")
    probability: float = Field(..., description="Model probability of heart disease (0 to 1)")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class InfoResponse(BaseModel):
    model_type: str
    n_features: int
    features: List[str]
    target: str
    training: Dict[str, Any]
