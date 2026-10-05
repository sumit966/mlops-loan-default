"""FastAPI inference service for loan default prediction."""
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="Loan Default Prediction API",
    description="MLOps pipeline serving loan default predictions",
    version="1.0.0",
)

MODEL = None
SCALER = None


class LoanApplication(BaseModel):
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=850)
    employment_years: int = Field(..., ge=0)
    num_previous_loans: int = Field(0, ge=0)
    has_mortgage: int = Field(0, ge=0, le=1)
    num_dependents: int = Field(0, ge=0)


class PredictionResponse(BaseModel):
    default_probability: float
    risk_level: str
    recommendation: str


@app.on_event("startup")
def load_model():
    global MODEL, SCALER
    try:
        MODEL = joblib.load("models/best_model.joblib")
        SCALER = joblib.load("models/scaler.joblib")
        print("[OK] Model loaded")
    except FileNotFoundError:
        print("[WARN] Model not found. Run: python src/train.py")


def build_features(app: LoanApplication) -> np.ndarray:
    loan_to_income = app.loan_amount / app.income
    credit_per_year = app.credit_score / (app.employment_years + 1)
    risk_score = max(700 - app.credit_score, 0) * 0.01 + loan_to_income * 0.5

    return np.array([[
        app.age, app.income, app.loan_amount, app.credit_score,
        app.employment_years, app.num_previous_loans,
        app.has_mortgage, app.num_dependents,
        loan_to_income, credit_per_year, risk_score
    ]])


@app.get("/")
def root():
    return {"message": "Loan Default Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = build_features(application)
    features_scaled = SCALER.transform(features)
    proba = float(MODEL.predict_proba(features_scaled)[0, 1])

    if proba < 0.3:
        risk, rec = "Low", "Approve"
    elif proba < 0.6:
        risk, rec = "Medium", "Approve with conditions"
    else:
        risk, rec = "High", "Reject or require collateral"

    return PredictionResponse(
        default_probability=round(proba, 4),
        risk_level=risk,
        recommendation=rec,
    )
