from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Cloud Cost Anomaly Detection API")

@app.get("/")
def root():
    return {"message": "Cloud Cost Anomaly API", "status": "online", "version": "1.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

class CostPayload(BaseModel):
    cost: float
    usage_hours: float
    cpu_utilization: float

model = joblib.load("models/enhanced_temporal_model.pkl")

@app.post("/detect-anomalies")
def detect(payload: CostPayload):
    cost_per_hour = payload.cost / payload.usage_hours
    cpu_cost_ratio = payload.cost / (payload.cpu_utilization + 1)

    features = np.array([[
        payload.cost,
        payload.usage_hours,
        payload.cpu_utilization,
        cost_per_hour,
        cpu_cost_ratio
    ]])

    is_anomaly = model.predict(features)[0] == -1

    if is_anomaly:
        explanation = "High cost with low utilization detected"
        recommendation = "Investigate idle resources and consider rightsizing"
    else:
        explanation = "Cost pattern appears normal"
        recommendation = "No action required"

    return {
        "is_anomaly": is_anomaly,
        "confidence": float(abs(model.decision_function(features)[0])),
        "explanation": explanation,
        "recommendation": recommendation,
        "cost_per_hour": round(cost_per_hour, 2),
        "risk_level": "HIGH" if is_anomaly else "LOW"
    }