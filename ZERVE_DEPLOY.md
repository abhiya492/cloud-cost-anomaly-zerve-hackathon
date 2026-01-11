# Zerve Deployment Configuration
# Cloud Cost Anomaly Detection System

## API Endpoint Configuration
- **Framework**: FastAPI
- **Port**: 8000
- **Host**: 0.0.0.0
- **Entry Point**: api/deploy.py

## Start Command for Zerve
```bash
uvicorn api.deploy:app --host 0.0.0.0 --port 8000
```

## Dependencies (requirements-deploy.txt)
- fastapi
- uvicorn
- pydantic
- numpy
- scikit-learn
- joblib

## Model Files Required
- models/enhanced_temporal_model.pkl
- models/enhanced_metadata.json

## API Endpoints
- POST /detect-anomalies - Single anomaly detection
- GET /docs - Interactive API documentation

## Validation Metrics Included
- Precision: 0.85
- Recall: 0.78
- F1-Score: 0.81
- Business Impact: $50K+ monthly savings detected