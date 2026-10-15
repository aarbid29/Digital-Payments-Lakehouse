import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
churn_model = joblib.load(ROOT / "models/churn_model.joblib")
FEATURES = json.loads((ROOT / "models/churn_features.json").read_text())
seg = joblib.load(ROOT / "models/segmentation.joblib")

# Update these if you retrain segmentation and the cluster ids change.
SEGMENT_NAMES = {0: "Regulars", 1: "Lapsed low-value",
                 2: "Champions", 3: "At-risk high-value"}

app = FastAPI(title="Payments ML API", version="1.0",
              description="Churn prediction and customer segmentation (synthetic data).")


class ChurnRequest(BaseModel):
    recency_days: float = Field(ge=0)
    tenure_days: float = Field(ge=0)
    freq_total: int = Field(ge=1)
    active_days: int = Field(ge=1)
    freq_30d: int = Field(ge=0)
    freq_60d: int = Field(ge=0)
    freq_90d: int = Field(ge=0)
    monetary_90d: float = Field(ge=0)
    avg_amount: float = Field(ge=0)
    n_merchants: int = Field(ge=0)
    transfer_share: float = Field(ge=0, le=1)
    kyc_level: int = Field(ge=1, le=3)

    model_config = {"json_schema_extra": {"examples": [{
        "recency_days": 70, "tenure_days": 150, "freq_total": 20, "active_days": 15,
        "freq_30d": 0, "freq_60d": 0, "freq_90d": 3, "monetary_90d": 3000,
        "avg_amount": 900, "n_merchants": 3, "transfer_share": 0.3, "kyc_level": 1}]}}


class ChurnResponse(BaseModel):
    churn_probability: float
    risk_band: str


class SegmentRequest(BaseModel):
    recency_days: float = Field(ge=0)
    frequency: int = Field(ge=1)
    monetary: float = Field(ge=0)

    model_config = {"json_schema_extra": {"examples": [
        {"recency_days": 5, "frequency": 44, "monetary": 38000}]}}


class SegmentResponse(BaseModel):
    segment_id: int
    segment_name: str


@app.get("/health")
def health():
    return {"status": "ok", "features": len(FEATURES)}


@app.post("/predict/churn", response_model=ChurnResponse)
def predict_churn(req: ChurnRequest):
    row = req.model_dump()
    row["trend_30_vs_90"] = row["freq_30d"] / (row["freq_90d"] / 3 + 1e-9)
    row["freq_per_active_day"] = row["freq_total"] / max(row["active_days"], 1)
    X = pd.DataFrame([row])[FEATURES]
    p = float(churn_model.predict_proba(X)[0, 1])
    band = "high" if p >= 0.6 else "medium" if p >= 0.3 else "low"
    return ChurnResponse(churn_probability=round(p, 4), risk_band=band)


@app.post("/segment", response_model=SegmentResponse)
def segment(req: SegmentRequest):
    X = pd.DataFrame({
        "recency": np.log1p([req.recency_days]),
        "frequency": np.log1p([req.frequency]),
        "monetary": np.log1p([req.monetary]),
    })
    label = int(seg["kmeans"].predict(seg["scaler"].transform(X))[0])
    return SegmentResponse(segment_id=label, segment_name=SEGMENT_NAMES[label])