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
jupyter notebook notebooks/day2_enhanced.ipynb
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

### Day 2: Enhanced ML Anomaly Detection
- Enhanced dataset with utilization, environment, and service context
- Domain-driven feature engineering (cost_per_hour, cpu_cost_ratio)
- Improved anomaly explainability with business logic
- Business-aligned anomaly definitions
- Isolation Forest with tuned parameters
- Model comparison showing significant improvements

**Run:** `jupyter notebook notebooks/day2_enhanced.ipynb`

#### Day 2 Enhancements
- **Better Data**: Realistic AWS-like schema with CPU utilization and environment context
- **Smart Features**: High-signal features that directly map to cloud inefficiency
- **Explainable**: Each anomaly comes with business reasoning
- **Validated**: Business logic ground truth for meaningful metrics

## Key Results

- **Detection Method**: Isolation Forest (unsupervised ML)
- **Features**: Cost lags, rolling statistics, change rates
- **Validation**: Precision/Recall/F1 metrics
- **Business Impact**: Estimated monthly cost savings
- **Model**: Saved for production deployment