import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os

class FeatureStore:
    """Centralized feature management and storage"""
    
    def __init__(self, storage_path: str = "features/"):
        self.storage_path = storage_path
        self.feature_registry = {}
        self._ensure_storage_path()
    
    def _ensure_storage_path(self):
        """Create storage directory if it doesn't exist"""
        os.makedirs(self.storage_path, exist_ok=True)
    
    def register_feature(self, name: str, description: str, feature_type: str, dependencies: List[str] = None):
        """Register a new feature in the feature store"""
        self.feature_registry[name] = {
            'description': description,
            'type': feature_type,
            'dependencies': dependencies or [],
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        self._save_registry()
    
    def get_features(self, account_id: str, date_range: Tuple[str, str], feature_names: List[str] = None) -> pd.DataFrame:
        """Get features for specific account and date range"""
        start_date, end_date = date_range
        
        # Load base data (in production, this would query a database)
        base_data = self._load_base_data(account_id, start_date, end_date)
        
        if base_data.empty:
            return pd.DataFrame()
        
        # Generate all features
        features_df = self._generate_features(base_data)
        
        # Filter requested features
        if feature_names:
            available_features = [f for f in feature_names if f in features_df.columns]
            features_df = features_df[available_features]
        
        return features_df
    
    def _load_base_data(self, account_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load base cost data (mock implementation)"""
        # In production, this would query your data warehouse
        # For demo, we'll generate sample data
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic cost data
        np.random.seed(42)
        base_cost = 50
        trend = 0.1
        seasonality = 10 * np.sin(2 * np.pi * np.arange(len(date_range)) / 7)  # Weekly pattern
        noise = np.random.normal(0, 5, len(date_range))
        
        costs = base_cost + trend * np.arange(len(date_range)) + seasonality + noise
        costs = np.maximum(costs, 10)  # Ensure positive costs
        
        data = {
            'date': date_range,
            'account_id': [account_id] * len(date_range),
            'cost': costs,
            'usage_hours': np.random.uniform(20, 24, len(date_range)),
            'cpu_utilization': np.random.uniform(30, 80, len(date_range)),
            'service': np.random.choice(['EC2', 'RDS', 'Lambda'], len(date_range)),
            'environment': np.random.choice(['prod', 'dev', 'staging'], len(date_range))
        }
        
        return pd.DataFrame(data)
    
    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive feature set"""
        features = df.copy()
        
        # Basic derived features
        features['cost_per_hour'] = features['cost'] / features['usage_hours']
        features['cpu_cost_ratio'] = features['cost'] / (features['cpu_utilization'] + 1)
        
        # Time-based features
        features['date'] = pd.to_datetime(features['date'])
        features['day_of_week'] = features['date'].dt.dayofweek
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['day_of_month'] = features['date'].dt.day
        features['month'] = features['date'].dt.month
        features['quarter'] = features['date'].dt.quarter
        
        # Rolling window features
        for window in [3, 7, 14, 30]:
            features[f'cost_rolling_mean_{window}d'] = features.groupby('account_id')['cost'].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
            features[f'cost_rolling_std_{window}d'] = features.groupby('account_id')['cost'].rolling(window, min_periods=1).std().reset_index(0, drop=True)
            features[f'cost_rolling_max_{window}d'] = features.groupby('account_id')['cost'].rolling(window, min_periods=1).max().reset_index(0, drop=True)
            features[f'cost_rolling_min_{window}d'] = features.groupby('account_id')['cost'].rolling(window, min_periods=1).min().reset_index(0, drop=True)
        
        # Lag features
        for lag in [1, 2, 3, 7]:
            features[f'cost_lag_{lag}d'] = features.groupby('account_id')['cost'].shift(lag)
            features[f'cpu_lag_{lag}d'] = features.groupby('account_id')['cpu_utilization'].shift(lag)
        
        # Change features
        features['cost_change_1d'] = features.groupby('account_id')['cost'].pct_change()
        features['cost_change_7d'] = features.groupby('account_id')['cost'].pct_change(7)
        features['cost_change_30d'] = features.groupby('account_id')['cost'].pct_change(30)
        
        # Volatility features
        features['cost_volatility_7d'] = features.groupby('account_id')['cost'].rolling(7, min_periods=1).std().reset_index(0, drop=True)
        features['cost_volatility_30d'] = features.groupby('account_id')['cost'].rolling(30, min_periods=1).std().reset_index(0, drop=True)
        
        # Efficiency features
        features['efficiency_score'] = features['cpu_utilization'] / (features['cost_per_hour'] + 1e-6)
        features['resource_utilization'] = features['usage_hours'] / 24.0
        
        # Categorical encoding
        features['service_encoded'] = pd.Categorical(features['service']).codes
        features['environment_encoded'] = pd.Categorical(features['environment']).codes
        
        return features
    
    def store_features(self, features_df: pd.DataFrame, feature_set_name: str):
        """Store computed features for later use"""
        file_path = os.path.join(self.storage_path, f"{feature_set_name}.parquet")
        features_df.to_parquet(file_path, index=False)
        
        # Update metadata
        metadata = {
            'feature_set_name': feature_set_name,
            'num_features': len(features_df.columns),
            'num_records': len(features_df),
            'created_at': datetime.now().isoformat(),
            'file_path': file_path
        }
        
        metadata_path = os.path.join(self.storage_path, f"{feature_set_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_features(self, feature_set_name: str) -> pd.DataFrame:
        """Load previously stored features"""
        file_path = os.path.join(self.storage_path, f"{feature_set_name}.parquet")
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        else:
            raise FileNotFoundError(f"Feature set '{feature_set_name}' not found")
    
    def list_feature_sets(self) -> List[Dict]:
        """List all available feature sets"""
        feature_sets = []
        for file in os.listdir(self.storage_path):
            if file.endswith('_metadata.json'):
                metadata_path = os.path.join(self.storage_path, file)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                feature_sets.append(metadata)
        return feature_sets
    
    def get_feature_importance(self, model_results: Dict) -> Dict[str, float]:
        """Calculate feature importance from model results"""
        # This would integrate with your ML models to get feature importance
        # For demo, return mock importance scores
        
        important_features = {
            'cost': 0.25,
            'cost_rolling_mean_7d': 0.20,
            'cost_change_1d': 0.15,
            'cpu_utilization': 0.12,
            'cost_per_hour': 0.10,
            'is_weekend': 0.08,
            'cost_volatility_7d': 0.06,
            'efficiency_score': 0.04
        }
        
        return important_features
    
    def _save_registry(self):
        """Save feature registry to disk"""
        registry_path = os.path.join(self.storage_path, "feature_registry.json")
        with open(registry_path, 'w') as f:
            json.dump(self.feature_registry, f, indent=2)
    
    def _load_registry(self):
        """Load feature registry from disk"""
        registry_path = os.path.join(self.storage_path, "feature_registry.json")
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                self.feature_registry = json.load(f)

# Feature engineering utilities
class FeatureEngineering:
    """Advanced feature engineering utilities"""
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features between feature pairs"""
        result_df = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplication interaction
                result_df[f"{feat1}_x_{feat2}"] = df[feat1] * df[feat2]
                
                # Division interaction (with safety check)
                result_df[f"{feat1}_div_{feat2}"] = df[feat1] / (df[feat2] + 1e-6)
        
        return result_df
    
    @staticmethod
    def create_polynomial_features(df: pd.DataFrame, features: List[str], degree: int = 2) -> pd.DataFrame:
        """Create polynomial features"""
        result_df = df.copy()
        
        for feature in features:
            if feature in df.columns:
                for d in range(2, degree + 1):
                    result_df[f"{feature}_pow_{d}"] = df[feature] ** d
        
        return result_df
    
    @staticmethod
    def create_binned_features(df: pd.DataFrame, feature: str, bins: int = 5) -> pd.DataFrame:
        """Create binned categorical features from continuous ones"""
        result_df = df.copy()
        
        if feature in df.columns:
            result_df[f"{feature}_binned"] = pd.cut(df[feature], bins=bins, labels=False)
        
        return result_df