import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class ModelMonitor:
    """Monitor model performance and detect drift over time"""
    
    def __init__(self, model_name: str = "default_model", storage_path: str = "monitoring/"):
        self.model_name = model_name
        self.storage_path = storage_path
        self.performance_history = []
        self.drift_history = []
        self._ensure_storage_path()
    
    def _ensure_storage_path(self):
        """Create monitoring directory if it doesn't exist"""
        os.makedirs(self.storage_path, exist_ok=True)
    
    def log_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       timestamp: datetime = None, metadata: Dict = None):
        """Log model performance metrics"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = (y_true == y_pred).mean()
        
        # Calculate additional metrics
        true_positive_rate = recall
        false_positive_rate = ((y_pred == 1) & (y_true == 0)).sum() / (y_true == 0).sum() if (y_true == 0).sum() > 0 else 0
        
        performance_record = {
            'timestamp': timestamp.isoformat(),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'true_positive_rate': float(true_positive_rate),
            'false_positive_rate': float(false_positive_rate),
            'num_samples': int(len(y_true)),
            'num_anomalies': int(y_true.sum()),
            'predicted_anomalies': int(y_pred.sum()),
            'metadata': metadata or {}
        }
        
        self.performance_history.append(performance_record)
        self._save_performance_history()
        
        return performance_record
    
    def detect_performance_drift(self, window_size: int = 30, threshold: float = 0.1) -> Dict:
        """Detect if model performance has degraded significantly"""
        if len(self.performance_history) < window_size * 2:
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        # Get recent and baseline performance
        recent_performance = self.performance_history[-window_size:]
        baseline_performance = self.performance_history[-window_size*2:-window_size]
        
        # Calculate average metrics
        recent_f1 = np.mean([p['f1_score'] for p in recent_performance])
        baseline_f1 = np.mean([p['f1_score'] for p in baseline_performance])
        
        recent_precision = np.mean([p['precision'] for p in recent_performance])
        baseline_precision = np.mean([p['precision'] for p in baseline_performance])
        
        # Check for significant degradation
        f1_degradation = (baseline_f1 - recent_f1) / (baseline_f1 + 1e-6)
        precision_degradation = (baseline_precision - recent_precision) / (baseline_precision + 1e-6)
        
        drift_detected = f1_degradation > threshold or precision_degradation > threshold
        
        drift_record = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': drift_detected,
            'f1_degradation': f1_degradation,
            'precision_degradation': precision_degradation,
            'recent_f1': recent_f1,
            'baseline_f1': baseline_f1,
            'recent_precision': recent_precision,
            'baseline_precision': baseline_precision,
            'threshold': threshold
        }
        
        if drift_detected:
            drift_record['recommendation'] = 'Model retraining recommended'
            self._log_drift_alert(drift_record)
        
        return drift_record
    
    def detect_data_drift(self, reference_data: pd.DataFrame, 
                         current_data: pd.DataFrame, 
                         features: List[str]) -> Dict:
        """Detect drift in input data distribution"""
        drift_results = {}
        
        for feature in features:
            if feature in reference_data.columns and feature in current_data.columns:
                # Use Kolmogorov-Smirnov test for continuous features
                ref_values = reference_data[feature].dropna()
                curr_values = current_data[feature].dropna()
                
                if len(ref_values) > 0 and len(curr_values) > 0:
                    ks_statistic, p_value = stats.ks_2samp(ref_values, curr_values)
                    
                    # Calculate distribution statistics
                    ref_mean, ref_std = ref_values.mean(), ref_values.std()
                    curr_mean, curr_std = curr_values.mean(), curr_values.std()
                    
                    mean_shift = abs(curr_mean - ref_mean) / (ref_std + 1e-6)
                    std_change = abs(curr_std - ref_std) / (ref_std + 1e-6)
                    
        drift_results[feature] = {
                        'ks_statistic': float(ks_statistic),
                        'p_value': float(p_value),
                        'drift_detected': bool(p_value < 0.05),  # 5% significance level
                        'mean_shift': float(mean_shift),
                        'std_change': float(std_change),
                        'ref_mean': float(ref_mean),
                        'curr_mean': float(curr_mean),
                        'ref_std': float(ref_std),
                        'curr_std': float(curr_std)
                    }
        
        # Overall drift assessment
        drifted_features = [f for f, r in drift_results.items() if r['drift_detected']]
        overall_drift = len(drifted_features) / len(features) if features else 0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_drift_ratio': float(overall_drift),
            'drifted_features': drifted_features,
            'total_features_checked': len(features),
            'drift_detected': bool(overall_drift > 0.3),  # 30% of features drifted
            'feature_results': drift_results
        }
        
        self.drift_history.append(summary)
        self._save_drift_history()
        
        return summary
    
    def get_performance_trend(self, metric: str = 'f1_score', days: int = 30) -> Dict:
        """Get performance trend over specified period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_records = [
            r for r in self.performance_history 
            if datetime.fromisoformat(r['timestamp']) >= cutoff_date
        ]
        
        if not recent_records:
            return {'trend': 'insufficient_data'}
        
        values = [r[metric] for r in recent_records]
        timestamps = [datetime.fromisoformat(r['timestamp']) for r in recent_records]
        
        # Calculate trend
        if len(values) > 1:
            # Simple linear regression for trend
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            trend_direction = 'improving' if slope > 0.001 else 'degrading' if slope < -0.001 else 'stable'
            
            return {
                'trend': trend_direction,
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'current_value': values[-1],
                'average_value': np.mean(values),
                'min_value': np.min(values),
                'max_value': np.max(values),
                'num_data_points': len(values)
            }
        else:
            return {'trend': 'insufficient_data'}
    
    def generate_monitoring_report(self) -> Dict:
        """Generate comprehensive monitoring report"""
        report = {
            'model_name': self.model_name,
            'report_timestamp': datetime.now().isoformat(),
            'performance_summary': {},
            'drift_summary': {},
            'recommendations': []
        }
        
        # Performance summary
        if self.performance_history:
            latest_performance = self.performance_history[-1]
            performance_trend = self.get_performance_trend()
            
            report['performance_summary'] = {
                'latest_metrics': latest_performance,
                'trend_analysis': performance_trend,
                'total_evaluations': len(self.performance_history)
            }
            
            # Performance-based recommendations
            if performance_trend['trend'] == 'degrading':
                report['recommendations'].append({
                    'type': 'performance_degradation',
                    'message': 'Model performance is degrading. Consider retraining.',
                    'priority': 'high'
                })
        
        # Drift summary
        if self.drift_history:
            latest_drift = self.drift_history[-1]
            report['drift_summary'] = latest_drift
            
            if latest_drift['drift_detected']:
                report['recommendations'].append({
                    'type': 'data_drift',
                    'message': f"Data drift detected in {len(latest_drift['drifted_features'])} features.",
                    'priority': 'medium',
                    'affected_features': latest_drift['drifted_features']
                })
        
        # General recommendations
        if len(self.performance_history) > 100:
            report['recommendations'].append({
                'type': 'maintenance',
                'message': 'Consider archiving old monitoring data to improve performance.',
                'priority': 'low'
            })
        
        return report
    
    def _log_drift_alert(self, drift_record: Dict):
        """Log drift alert for immediate attention"""
        alert_file = os.path.join(self.storage_path, f"{self.model_name}_drift_alerts.log")
        
        alert_message = f"""
        DRIFT ALERT - {drift_record['timestamp']}
        Model: {self.model_name}
        F1 Degradation: {drift_record['f1_degradation']:.3f}
        Precision Degradation: {drift_record['precision_degradation']:.3f}
        Recommendation: {drift_record.get('recommendation', 'Review model performance')}
        """
        
        with open(alert_file, 'a') as f:
            f.write(alert_message + '\n')
    
    def _save_performance_history(self):
        """Save performance history to disk"""
        file_path = os.path.join(self.storage_path, f"{self.model_name}_performance.json")
        with open(file_path, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
    
    def _save_drift_history(self):
        """Save drift history to disk"""
        file_path = os.path.join(self.storage_path, f"{self.model_name}_drift.json")
        with open(file_path, 'w') as f:
            json.dump(self.drift_history, f, indent=2)
    
    def load_history(self):
        """Load monitoring history from disk"""
        # Load performance history
        perf_file = os.path.join(self.storage_path, f"{self.model_name}_performance.json")
        if os.path.exists(perf_file):
            with open(perf_file, 'r') as f:
                self.performance_history = json.load(f)
        
        # Load drift history
        drift_file = os.path.join(self.storage_path, f"{self.model_name}_drift.json")
        if os.path.exists(drift_file):
            with open(drift_file, 'r') as f:
                self.drift_history = json.load(f)

class AutoRetrainer:
    """Automatic model retraining based on monitoring results"""
    
    def __init__(self, model_monitor: ModelMonitor, retrain_threshold: float = 0.15):
        self.monitor = model_monitor
        self.retrain_threshold = retrain_threshold
        self.retrain_history = []
    
    def should_retrain(self) -> Tuple[bool, str]:
        """Determine if model should be retrained"""
        # Check performance drift
        drift_result = self.monitor.detect_performance_drift(threshold=self.retrain_threshold)
        
        if drift_result['drift_detected']:
            return True, f"Performance degradation detected: F1 degraded by {drift_result['f1_degradation']:.3f}"
        
        # Check data drift
        if self.monitor.drift_history:
            latest_drift = self.monitor.drift_history[-1]
            if latest_drift['drift_detected'] and latest_drift['overall_drift_ratio'] > 0.5:
                return True, f"Significant data drift detected in {len(latest_drift['drifted_features'])} features"
        
        # Check time since last retrain
        if self.retrain_history:
            last_retrain = datetime.fromisoformat(self.retrain_history[-1]['timestamp'])
            days_since_retrain = (datetime.now() - last_retrain).days
            
            if days_since_retrain > 90:  # 3 months
                return True, "Scheduled retraining: 90 days since last retrain"
        
        return False, "No retraining needed"
    
    def trigger_retrain(self, reason: str) -> Dict:
        """Trigger model retraining process"""
        retrain_record = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'status': 'triggered',
            'model_name': self.monitor.model_name
        }
        
        self.retrain_history.append(retrain_record)
        
        # In production, this would trigger your ML pipeline
        print(f"RETRAIN TRIGGERED: {reason}")
        
        return retrain_record