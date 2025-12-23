import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class CostOptimizationEngine:
    """Generate specific cost optimization recommendations"""
    
    def __init__(self):
        self.reserved_discount = 0.4  # 40% discount for Reserved Instances
        self.rightsizing_threshold = 0.3  # 30% CPU utilization threshold
    
    def analyze_cost_data(self, df: pd.DataFrame) -> List[Dict]:
        """Generate optimization recommendations from cost data"""
        recommendations = []
        
        # Reserved Instance recommendations
        recommendations.extend(self._recommend_reserved_instances(df))
        
        # Rightsizing recommendations
        recommendations.extend(self._recommend_rightsizing(df))
        
        # Unused resource recommendations
        recommendations.extend(self._recommend_unused_cleanup(df))
        
        # Scheduling optimizations
        recommendations.extend(self._recommend_scheduling_optimization(df))
        
        # Storage optimizations
        recommendations.extend(self._recommend_storage_optimization(df))
        
        # Sort by potential savings
        recommendations.sort(key=lambda x: x['monthly_savings'], reverse=True)
        
        return recommendations
    
    def _recommend_reserved_instances(self, df: pd.DataFrame) -> List[Dict]:
        """Recommend Reserved Instances for consistent workloads"""
        recommendations = []
        
        if 'service' in df.columns and 'cost' in df.columns:
            service_costs = df.groupby('service')['cost'].agg(['sum', 'count']).reset_index()
            
            for _, row in service_costs.iterrows():
                service = row['service']
                monthly_cost = row['sum']
                consistency = row['count'] / len(df)
                
                if consistency > 0.8 and monthly_cost > 100:
                    savings = monthly_cost * self.reserved_discount
                    
                    # Service-specific recommendations
                    if 'EC2' in service:
                        desc = f'Switch EC2 instances to Reserved: Save ${savings:.0f}/month'
                    elif 'RDS' in service:
                        desc = f'Purchase RDS Reserved capacity: Save ${savings:.0f}/month'
                    elif 'Lambda' in service:
                        desc = f'Use Lambda Provisioned Concurrency: Save ${savings:.0f}/month'
                    else:
                        desc = f'Switch {service} to Reserved pricing: Save ${savings:.0f}/month'
                    
                    recommendations.append({
                        'type': 'Reserved Instance',
                        'service': service,
                        'description': desc,
                        'monthly_savings': savings,
                        'annual_savings': savings * 12,
                        'confidence': min(consistency, 0.95)
                    })
        
        return recommendations
    
    def _recommend_rightsizing(self, df: pd.DataFrame) -> List[Dict]:
        """Recommend instance rightsizing based on utilization"""
        recommendations = []
        
        if 'cpu_utilization' in df.columns and 'cost' in df.columns:
            underutilized = df[df['cpu_utilization'] < self.rightsizing_threshold]
            
            for _, row in underutilized.iterrows():
                current_cost = row['cost']
                cpu_util = row['cpu_utilization']
                service = row.get('service', 'Unknown')
                
                if current_cost > 50:
                    reduction_factor = 0.4 if cpu_util < 0.15 else 0.3
                    savings = current_cost * reduction_factor
                    
                    # Instance-specific rightsizing
                    if 'EC2' in service:
                        desc = f'Rightsize EC2 t3.large to t3.medium: Save ${savings:.0f}/month'
                    elif 'RDS' in service:
                        desc = f'Downsize RDS db.m5.large to db.t3.medium: Save ${savings:.0f}/month'
                    elif 'Lambda' in service:
                        desc = f'Reduce Lambda memory allocation: Save ${savings:.0f}/month'
                    else:
                        desc = f'Optimize {service} resource allocation: Save ${savings:.0f}/month'
                    
                    recommendations.append({
                        'type': 'Rightsizing',
                        'service': service,
                        'description': desc,
                        'monthly_savings': savings,
                        'annual_savings': savings * 12,
                        'confidence': 0.8 if cpu_util < 0.15 else 0.6
                    })
        
        return recommendations
    
    def _recommend_unused_cleanup(self, df: pd.DataFrame) -> List[Dict]:
        """Recommend cleanup of unused resources"""
        recommendations = []
        
        if 'usage_hours' in df.columns and 'cost' in df.columns:
            unused = df[df['usage_hours'] < 24]
            
            # Limit to top 3 to avoid repetition
            unused = unused.nlargest(3, 'cost')
            
            for _, row in unused.iterrows():
                cost = row['cost']
                usage = row['usage_hours']
                
                if cost > 10:
                    savings = cost * 0.9
                    recommendations.append({
                        'type': 'Resource Cleanup',
                        'service': row.get('service', 'Unknown'),
                        'description': f'Remove unused {row.get("service", "resource")}: Save ${savings:.0f}/month',
                        'monthly_savings': savings,
                        'annual_savings': savings * 12,
                        'confidence': 0.9
                    })
        
        return recommendations
    
    def _recommend_scheduling_optimization(self, df: pd.DataFrame) -> List[Dict]:
        """Recommend scheduling optimizations"""
        recommendations = []
        
        if 'environment' in df.columns and 'cost' in df.columns:
            # Dev environment optimization
            dev_costs = df[df['environment'] == 'dev']['cost'].sum()
            if dev_costs > 50:
                savings = dev_costs * 0.6  # 60% savings from scheduling
                recommendations.append({
                    'type': 'Scheduling',
                    'service': 'Dev Environment',
                    'description': f'Auto-shutdown dev resources 6PM-8AM: Save ${savings:.0f}/month',
                    'monthly_savings': savings,
                    'annual_savings': savings * 12,
                    'confidence': 0.85
                })
        
        return recommendations
    
    def _recommend_storage_optimization(self, df: pd.DataFrame) -> List[Dict]:
        """Recommend storage tier optimizations"""
        recommendations = []
        
        if 'service' in df.columns and 'cost' in df.columns:
            storage_services = df[df['service'].str.contains('S3|Storage', na=False)]
            
            for _, row in storage_services.iterrows():
                cost = row['cost']
                if cost > 30:
                    savings = cost * 0.25  # 25% savings from intelligent tiering
                    recommendations.append({
                        'type': 'Storage Optimization',
                        'service': row['service'],
                        'description': f'Enable S3 Intelligent Tiering: Save ${savings:.0f}/month',
                        'monthly_savings': savings,
                        'annual_savings': savings * 12,
                        'confidence': 0.75
                    })
        
        return recommendations

class MultiCloudAnalyzer:
    """Analyze costs across AWS, Azure, and GCP"""
    
    def __init__(self):
        self.cloud_patterns = {
            'aws': {'anomaly_threshold': 2.0, 'services': ['EC2', 'S3', 'RDS']},
            'azure': {'anomaly_threshold': 2.2, 'services': ['VM', 'Storage', 'SQL']},
            'gcp': {'anomaly_threshold': 1.8, 'services': ['Compute', 'Storage', 'SQL']}
        }
    
    def detect_cloud_anomalies(self, df: pd.DataFrame, cloud: str) -> List[Dict]:
        """Detect cloud-specific anomalies"""
        if cloud not in self.cloud_patterns or df.empty:
            return []
        
        threshold = self.cloud_patterns[cloud]['anomaly_threshold']
        anomalies = []
        
        if 'cost' in df.columns:
            mean_cost = df['cost'].mean()
            std_cost = df['cost'].std()
            
            cloud_anomalies = df[df['cost'] > mean_cost + threshold * std_cost]
            
            for _, row in cloud_anomalies.iterrows():
                anomalies.append({
                    'cloud': cloud,
                    'service': row.get('service', 'Unknown'),
                    'cost': row['cost'],
                    'anomaly_score': (row['cost'] - mean_cost) / std_cost if std_cost > 0 else 0
                })
        
        return anomalies