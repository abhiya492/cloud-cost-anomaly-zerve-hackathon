import requests
import json

# Test data
test_data = {
    "date": "2025-10-15",
    "account_id": "acct-1",
    "service": "EC2",
    "environment": "prod",
    "usage_hours": 24,
    "cpu_utilization": 15,
    "cost": 160
}

# Test API
try:
    # Health check
    health = requests.get("http://localhost:8000/health")
    print(f"Health: {health.json()}")
    
    # Single detection
    response = requests.post("http://localhost:8000/detect-anomalies", json=test_data)
    print(f"Detection: {response.json()}")
    
    # Batch detection
    batch_response = requests.post("http://localhost:8000/batch-detect", json=[test_data, test_data])
    print(f"Batch: {batch_response.json()}")
    
except Exception as e:
    print(f"Error: {e}")
    print("Make sure API is running: uvicorn api.main:app --reload")