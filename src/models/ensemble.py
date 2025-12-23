import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnsembleAnomalyDetector:
    """Advanced ensemble model combining multiple ML approaches"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(n_estimators=200, contamination=0.15, random_state=42)
        self.scaler = MinMaxScaler()
        self.models = {}
        self.weights = {'isolation_forest': 0.4, 'statistical': 0.3, 'trend': 0.3}
        
    def fit(self, df: pd.DataFrame):
        """Train ensemble models"""
        # Prepare features
        features = self._extract_features(df)
        X = features.fillna(0)
        
        # Train Isolation Forest
        self.isolation_forest.fit(X)
        
        # Train statistical model
        self._fit_statistical_model(df)
        
        # Train trend model
        self._fit_trend_model(df)
        
        return self
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """Predict anomalies using ensemble approach"""
        features = self._extract_features(df)
        X = features.fillna(0)
        
        # Get predictions from each model
        iso_pred = self.isolation_forest.predict(X)
        iso_scores = self.isolation_forest.decision_function(X)
        
        stat_pred = self._predict_statistical(df)
        trend_pred = self._predict_trend(df)
        
        # Ensemble voting
        ensemble_scores = (
            self.weights['isolation_forest'] * np.abs(iso_scores) +
            self.weights['statistical'] * stat_pred +
            self.weights['trend'] * trend_pred
        )
        
        # Final predictions
        threshold = np.percentile(ensemble_scores, 85)
        final_predictions = ensemble_scores > threshold
        
        return {
            'is_anomaly': final_predictions,
            'confidence': np.clip(ensemble_scores / ensemble_scores.max(), 0, 1),
            'isolation_forest_score': iso_scores,
            'statistical_score': stat_pred,
            'trend_score': trend_pred,
            'ensemble_score': ensemble_scores
        }
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive features"""
        features = df.copy()
        
        # Basic features
        features['cost_per_hour'] = features['cost'] / features['usage_hours']
        features['cpu_cost_ratio'] = features['cost'] / (features['cpu_utilization'] + 1)
        features['is_weekend'] = pd.to_datetime(features['date']).dt.dayofweek >= 5
        
        # Time-based features
        features['hour'] = pd.to_datetime(features['date']).dt.hour
        features['day_of_month'] = pd.to_datetime(features['date']).dt.day
        features['month'] = pd.to_datetime(features['date']).dt.month
        
        # Rolling features by account
        for window in [3, 7, 14]:
            features[f'cost_rolling_mean_{window}d'] = features.groupby('account_id')['cost'].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
            features[f'cost_rolling_std_{window}d'] = features.groupby('account_id')['cost'].rolling(window, min_periods=1).std().reset_index(0, drop=True)
        
        # Lag features
        for lag in [1, 2, 3]:
            features[f'cost_lag_{lag}d'] = features.groupby('account_id')['cost'].shift(lag)
        
        # Change features
        features['cost_change_1d'] = features.groupby('account_id')['cost'].pct_change()
        features['cost_change_7d'] = features.groupby('account_id')['cost'].pct_change(7)
        
        # Select numeric features for ML
        numeric_features = [
            'cost', 'usage_hours', 'cpu_utilization', 'cost_per_hour', 'cpu_cost_ratio',
            'hour', 'day_of_month', 'month', 'cost_change_1d'
        ]
        
        return features[numeric_features]
    
    def _fit_statistical_model(self, df: pd.DataFrame):
        """Fit statistical anomaly detection"""
        # Z-score based detection per account
        self.stat_params = {}
        for account in df['account_id'].unique():
            account_data = df[df['account_id'] == account]['cost']
            self.stat_params[account] = {
                'mean': account_data.mean(),
                'std': account_data.std()
            }
    
    def _predict_statistical(self, df: pd.DataFrame) -> np.ndarray:
        """Statistical anomaly scores"""
        scores = np.zeros(len(df))
        for i, row in df.iterrows():
            account = row['account_id']
            if account in self.stat_params:
                params = self.stat_params[account]
                z_score = abs((row['cost'] - params['mean']) / (params['std'] + 1e-6))
                scores[i] = min(z_score / 3, 1.0)  # Normalize to 0-1
        return scores
    
    def _fit_trend_model(self, df: pd.DataFrame):
        """Fit trend-based detection"""
        # Simple trend detection based on recent changes
        self.trend_params = {}
        for account in df['account_id'].unique():
            account_data = df[df['account_id'] == account].sort_values('date')
            if len(account_data) > 3:
                recent_trend = account_data['cost'].tail(3).mean() - account_data['cost'].head(3).mean()
                self.trend_params[account] = {
                    'trend': recent_trend,
                    'volatility': account_data['cost'].std()
                }
    
    def _predict_trend(self, df: pd.DataFrame) -> np.ndarray:
        """Trend-based anomaly scores"""
        scores = np.zeros(len(df))
        for i, row in df.iterrows():
            account = row['account_id']
            if account in self.trend_params:
                params = self.trend_params[account]
                # Score based on deviation from expected trend
                expected_change = params['trend'] / 30  # Daily expected change
                actual_cost = row['cost']
                baseline_cost = 50  # Simplified baseline
                deviation = abs(actual_cost - baseline_cost - expected_change)
                scores[i] = min(deviation / (params['volatility'] + 1e-6) / 3, 1.0)
        return scores
    
    def save_model(self, path: str):
        """Save ensemble model"""
        model_data = {
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'stat_params': self.stat_params,
            'trend_params': self.trend_params,
            'weights': self.weights
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str):
        """Load ensemble model"""
        model_data = joblib.load(path)
        self.isolation_forest = model_data['isolation_forest']
        self.scaler = model_data['scaler']
        self.stat_params = model_data['stat_params']
        self.trend_params = model_data['trend_params']
        self.weights = model_data['weights']
        return self

# Prophet-like seasonality detection (simplified)
class SeasonalityDetector:
    """Simplified seasonality detection for cost patterns"""
    
    def __init__(self):
        self.seasonal_patterns = {}
    
    def fit(self, df: pd.DataFrame):
        """Learn seasonal patterns"""
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['hour'] = df['date'].dt.hour
        
        # Learn weekly patterns
        weekly_pattern = df.groupby('day_of_week')['cost'].mean()
        
        # Learn daily patterns (if hourly data available)
        daily_pattern = df.groupby('hour')['cost'].mean() if 'hour' in df.columns else None
        
        self.seasonal_patterns = {
            'weekly': weekly_pattern,
            'daily': daily_pattern
        }
        
        return self
    
    def predict_seasonal_baseline(self, df: pd.DataFrame) -> np.ndarray:
        """Predict expected cost based on seasonality"""
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        
        baseline = np.zeros(len(df))
        
        for i, row in df.iterrows():
            dow = row['day_of_week']
            if dow in self.seasonal_patterns['weekly']:
                baseline[i] = self.seasonal_patterns['weekly'][dow]
            else:
                baseline[i] = self.seasonal_patterns['weekly'].mean()
        
        return baseline
    
    def detect_seasonal_anomalies(self, df: pd.DataFrame, threshold: float = 2.0) -> np.ndarray:
        """Detect anomalies based on seasonal expectations"""
        baseline = self.predict_seasonal_baseline(df)
        deviations = np.abs(df['cost'].values - baseline) / (baseline + 1e-6)
        return deviations > threshold