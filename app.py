# app.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any
import joblib
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import uuid

# --- Global variables ---
model = None
scaler = None
le_gender = None
le_education = None
le_employment = None
le_purpose = None
le_region = None

GENDER_CLASSES = []
EDUCATION_CLASSES = []
EMPLOYMENT_CLASSES = []
PURPOSE_CLASSES = []
REGION_CLASSES = []

# --- Setup ---
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up Delinquency Risk Prediction API...")
    global model, scaler, le_gender, le_education, le_employment, le_purpose, le_region
    global GENDER_CLASSES, EDUCATION_CLASSES, EMPLOYMENT_CLASSES, PURPOSE_CLASSES, REGION_CLASSES

    try:
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

        model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

        le_gender = joblib.load(os.path.join(MODELS_DIR, "le_gender.pkl"))
        le_education = joblib.load(os.path.join(MODELS_DIR, "le_education.pkl"))
        le_employment = joblib.load(os.path.join(MODELS_DIR, "le_employment.pkl"))
        le_purpose = joblib.load(os.path.join(MODELS_DIR, "le_purpose.pkl"))
        le_region = joblib.load(os.path.join(MODELS_DIR, "le_region.pkl"))

        GENDER_CLASSES = le_gender.classes_.tolist()
        EDUCATION_CLASSES = le_education.classes_.tolist()
        EMPLOYMENT_CLASSES = le_employment.classes_.tolist()
        PURPOSE_CLASSES = le_purpose.classes_.tolist()
        REGION_CLASSES = le_region.classes_.tolist()

        logger.info("✅ Model and encoders loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load model or encoders: {e}")
        raise e

    yield

    logger.info("🛑 Shutting down Delinquency Risk Prediction API...")

# --- Initialize App ---
app = FastAPI(
    title="Delinquency Risk Prediction API",
    description="Predicts loan delinquency risk using a trained Random Forest model.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Input Schema ---
class BorrowerData(BaseModel):
    age: int = Field(..., ge=18, le=100)
    gender: str
    education: str
    employment_status: str
    monthly_income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    loan_tenure_months: int = Field(..., ge=6, le=36)
    loan_purpose: str
    region: str
    previous_default: int = Field(..., ge=0, le=1)
    num_previous_loans: int = Field(..., ge=0, le=50)
    credit_score: int = Field(..., ge=300, le=850)

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v not in GENDER_CLASSES:
            raise ValueError(f"Gender must be one of {GENDER_CLASSES}")
        return v

    @field_validator('education')
    @classmethod
    def validate_education(cls, v):
        if v not in EDUCATION_CLASSES:
            raise ValueError(f"Education must be one of {EDUCATION_CLASSES}")
        return v

    @field_validator('employment_status')
    @classmethod
    def validate_employment(cls, v):
        if v not in EMPLOYMENT_CLASSES:
            raise ValueError(f"Employment status must be one of {EMPLOYMENT_CLASSES}")
        return v

    @field_validator('loan_purpose')
    @classmethod
    def validate_loan_purpose(cls, v):
        if v not in PURPOSE_CLASSES:
            raise ValueError(f"Loan purpose must be one of {PURPOSE_CLASSES}")
        return v

    @field_validator('region')
    @classmethod
    def validate_region(cls, v):
        if v not in REGION_CLASSES:
            raise ValueError(f"Region must be one of {REGION_CLASSES}")
        return v

# --- Response Model ---
class PredictionResponse(BaseModel):
    request_id: str
    timestamp: str
    delinquency_risk: float
    risk_level: str
    model_version: str = "random_forest_v1"

# --- Prediction Endpoint ---
@app.post("/predict", response_model=PredictionResponse)
def predict_delinquency(data: BorrowerData):
    request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    logger.info(f"Received prediction request {request_id}: {data}")

    try:
        df = pd.DataFrame([data.dict()])

        df['gender'] = le_gender.transform(df['gender'])
        df['education'] = le_education.transform(df['education'])
        df['employment_status'] = le_employment.transform(df['employment_status'])
        df['loan_purpose'] = le_purpose.transform(df['loan_purpose'])
        df['region'] = le_region.transform(df['region'])

        df['income_to_loan_ratio'] = df['monthly_income'] / df['loan_amount']
        df['default_history_weight'] = df['previous_default'] * df['num_previous_loans']

        feature_cols = [
            'age', 'gender', 'education', 'employment_status', 'monthly_income',
            'loan_amount', 'loan_tenure_months', 'loan_purpose', 'region',
            'previous_default', 'num_previous_loans', 'credit_score',
            'income_to_loan_ratio', 'default_history_weight'
        ]
        df = df[feature_cols]

        risk_prob = model.predict_proba(df)[0][1]
        risk_prob = float(risk_prob)

        risk_level = "High" if risk_prob >= 0.7 else "Medium" if risk_prob >= 0.4 else "Low"

        response = {
            "request_id": request_id,
            "timestamp": timestamp,
            "delinquency_risk": round(risk_prob, 4),
            "risk_level": risk_level,
            "model_version": "random_forest_v1"
        }

        logger.info(f"Prediction success {request_id}: {response}")
        return response

    except ValueError as ve:
        msg = f"Invalid input: {str(ve)}"
        logger.error(f"Validation error {request_id}: {msg}")
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        msg = f"Internal prediction error: {str(e)}"
        logger.error(f"Server error {request_id}: {msg}")
        raise HTTPException(status_code=500, detail=msg)

# --- Health Check ---
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }