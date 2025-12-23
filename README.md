# Cloud Cost Anomaly Detection System

## Problem
Unexpected cloud cost spikes cause budget overruns.

## Solution
A time-series anomaly detection system that detects abnormal cloud spend and provides actionable recommendations.

## Quick Start

```bash
# Clone repository
git clone https://github.com/abhiya492/cloud-cost-anomaly-zerve-hackathon.git
cd cloud-cost-anomaly-zerve-hackathon

# Install dependencies
pip3 install -r requirements.txt

# Run Day 1 baseline analysis
jupyter notebook notebooks/day1_baseline.ipynb

# Run Day 2 enhanced ML analysis
jupyter notebook notebooks/day2_enhanced_temporal.ipynb

# Launch interactive dashboard
streamlit run dashboard/app.py
```

## Project Structure

```
cloud-cost-anomaly/
├── data/                    # Cost data files
├── notebooks/              # Analysis notebooks
├── models/                 # Trained ML models
├── api/                    # API endpoints (Day 3)
├── src/                    # Source code
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Implementation Progress

### Day 1: Baseline Anomaly Detection
- Dataset created (synthetic, realistic)
- Baseline anomaly detection using rolling statistics
- Visual validation of anomalies
- Initial recommendation logic

**Run:** `jupyter notebook notebooks/day1_baseline.ipynb`

### Day 2: ML Anomaly Detection
- Implemented Isolation Forest ML anomaly detection
- Feature engineering for time-series behavior
- Backtesting with precision, recall, F1-score
- Business KPI: estimated cost leakage detection
- Model comparison (baseline vs ML)
- Model saved for API deployment

**Run:** `jupyter notebook notebooks/day2_isolation_forest.ipynb`

### Day 3: Production API & Deployment
- FastAPI service with ML model integration
- RESTful endpoints for anomaly detection (/detect-anomalies, /batch-detect)
- Real-time alerting system for high-confidence anomalies
- Docker containerization for one-click deployment
- Production-ready error handling and logging
- Interactive API documentation (Swagger UI)

**Run API:**
```bash
# Start the API server
uvicorn api.main:app --reload

# Test the API
python api/test_api.py

# View interactive docs
open http://127.0.0.1:8000/docs

# Docker deployment
docker build -t cost-anomaly .
docker run -p 8000:8000 cost-anomaly
```

## Key Results

- **Detection Method**: Isolation Forest (unsupervised ML)
- **Features**: Cost lags, rolling statistics, change rates
- **Validation**: Precision/Recall/F1 metrics
- **Business Impact**: Estimated monthly cost savings
- **API**: Production-ready FastAPI service
- **Deployment**: Docker containerized

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /detect-anomalies` - Single anomaly detection
- `POST /batch-detect` - Batch anomaly detection
- `GET /docs` - Interactive API documentation