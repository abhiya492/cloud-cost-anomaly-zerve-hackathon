from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os
from datetime import datetime

app = FastAPI(title="Cloud Cost Anomaly Detection API", version="1.0.0")

# Load model
model = None
try:
    model = joblib.load("models/enhanced_temporal_model.pkl")
except:
    # Fallback model
    model = IsolationForest(n_estimators=200, contamination=0.15, random_state=42)

class CostData(BaseModel):
    date: str
    account_id: str
    service: str
    environment: str
    usage_hours: float
    cpu_utilization: float
    cost: float

class AnomalyResponse(BaseModel):
    is_anomaly: bool
    confidence: float
    anomaly_type: str
    recommendation: str

@app.get("/")
def root():
    return {"message": "Cloud Cost Anomaly Detection API", "status": "online"}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/detect-anomalies", response_model=AnomalyResponse)
def detect_anomalies(data: CostData):
    try:
        # Feature engineering
        cost_per_hour = data.cost / data.usage_hours
        cpu_cost_ratio = data.cost / (data.cpu_utilization + 1)
        is_weekend = pd.to_datetime(data.date).dayofweek >= 5
        
        features = np.array([[
            data.cost, data.usage_hours, data.cpu_utilization,
            cost_per_hour, cpu_cost_ratio, 0, int(is_weekend)
        ]])
        
        # Predict
        prediction = model.predict(features)[0]
        is_anomaly = prediction == -1
        
        # Confidence
        decision_score = model.decision_function(features)[0]
        confidence = min(1.0, abs(decision_score) * 2)
        
        # Classification
        if data.cpu_utilization < 20 and data.cost > 100:
            anomaly_type = "Idle Resource Waste"
            recommendation = "Consider rightsizing or auto-scaling"
        elif data.environment == "dev" and data.usage_hours > 20:
            anomaly_type = "Dev Resource Misuse"
            recommendation = "Implement scheduled shutdown"
        else:
            anomaly_type = "Unusual Cost Pattern"
            recommendation = "Monitor for recurring patterns"
        
        # Send alert if high confidence
        if is_anomaly and confidence > 0.8:
            send_alert(data, confidence, anomaly_type)
        
        return AnomalyResponse(
            is_anomaly=is_anomaly,
            confidence=confidence,
            anomaly_type=anomaly_type,
            recommendation=recommendation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-detect")
def batch_detect(data_list: List[CostData]):
    results = []
    for data in data_list:
        result = detect_anomalies(data)
        results.append(result.dict())
    return {"results": results, "count": len(results)}

def send_alert(data: CostData, confidence: float, anomaly_type: str):
    """Send alert for high-confidence anomalies"""
    alert_message = f"""
    🚨 HIGH CONFIDENCE ANOMALY DETECTED
    
    Account: {data.account_id}
    Service: {data.service}
    Cost: ${data.cost:.2f}
    Confidence: {confidence:.3f}
    Type: {anomaly_type}
    
    Immediate action required!
    """
    
    # In production, integrate with Slack/email
    print(f"ALERT SENT: {alert_message}")
    
    # Log to file
    with open("alerts.log", "a") as f:
        f.write(f"{datetime.now()}: {alert_message}\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)