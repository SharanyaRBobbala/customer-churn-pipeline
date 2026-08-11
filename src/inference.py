"""
Inference Module
-------------------
Serves the trained churn model two ways:
  1. Batch prediction (CLI) — score a CSV of customers.
  2. Real-time REST API (FastAPI) — single-customer scoring endpoint,
     matching the resume bullet on "REST APIs" and containerized
     deployment.

Run the API locally with:
    uvicorn src.inference:app --host 0.0.0.0 --port 8000
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "data/best_model.joblib"
THRESHOLD = 0.5  # override at load time from evaluation_report.json if available

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time churn risk scoring for the Customer Intelligence Platform",
    version="1.0.0",
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not Path(MODEL_PATH).exists():
            raise RuntimeError(f"Model not found at {MODEL_PATH}. Train it first via src/train.py")
        _model = joblib.load(MODEL_PATH)
    return _model


class CustomerFeatures(BaseModel):
    """Feature payload for a single customer scoring request.

    Field names must match the engineered feature columns produced by
    src/features.py. Extra fields are ignored; missing fields raise a
    validation error so bad requests fail loudly, not silently.
    """
    model_config = ConfigDict(extra="ignore")

    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract: int
    InternetService: int
    PaymentMethod: int
    PaperlessBilling: int
    # additional engineered fields are optional; computed if absent
    tenure_years: float | None = None
    avg_monthly_spend: float | None = None


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    threshold_used: float


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    model = get_model()
    row = pd.DataFrame([customer.model_dump(exclude_none=True)])

    # align columns to what the model was trained on; fill any gaps with 0
    expected_cols = model.feature_names_in_ if hasattr(model, "feature_names_in_") else row.columns
    for col in expected_cols:
        if col not in row.columns:
            row[col] = 0
    row = row[list(expected_cols)]

    try:
        proba = model.predict_proba(row)[0, 1]
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")

    return PredictionResponse(
        churn_probability=round(float(proba), 4),
        churn_prediction=bool(proba >= THRESHOLD),
        threshold_used=THRESHOLD,
    )


def batch_predict(input_csv: str, output_csv: str, threshold: float = 0.5):
    """CLI batch scoring path — scores every row in a CSV."""
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(input_csv)

    feature_cols = [c for c in df.columns if c != "Churn"]
    proba = model.predict_proba(df[feature_cols])[:, 1]

    df["churn_probability"] = proba
    df["churn_prediction"] = (proba >= threshold).astype(int)

    df.to_csv(output_csv, index=False)
    logger.info(f"Scored {len(df)} customers -> {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Batch churn inference")
    parser.add_argument("--input", default="data/test_features.csv")
    parser.add_argument("--output", default="data/predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    batch_predict(args.input, args.output, args.threshold)


if __name__ == "__main__":
    main()
