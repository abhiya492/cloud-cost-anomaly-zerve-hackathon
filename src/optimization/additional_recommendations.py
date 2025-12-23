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