import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

class AdvancedVisualizations:
    """Create 3D cost surfaces and interactive visualizations"""
    
    def create_3d_cost_surface(self, df: pd.DataFrame) -> go.Figure:
        """Create 3D cost surface visualization"""
        if df.empty or 'cost' not in df.columns:
            # Create demo data if empty
            x = np.arange(10)
            y = np.random.uniform(50, 200, 10)
            z = np.random.uniform(0.2, 0.8, 10)
        else:
            x = np.arange(len(df))
            y = df['cost'].values
            z = df.get('cpu_utilization', np.random.uniform(0.2, 0.8, len(df))).values
        
        # Create 3D scatter plot instead of surface for better visualization
        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y, 
            z=z,
            mode='markers+lines',
            marker=dict(
                size=8,
                color=y,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Cost ($)")
            ),
            line=dict(color='darkblue', width=4),
            name='Cost Surface'
        )])
        
        fig.update_layout(
            title='3D Cost Analysis: Time vs Cost vs CPU',
            scene=dict(
                xaxis_title='Time Period',
                yaxis_title='Cost ($)',
                zaxis_title='CPU Utilization',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800, height=500
        )
        
        return fig
    
    def create_network_graph(self, df: pd.DataFrame) -> go.Figure:
        """Create interactive network graph of services"""
        if df.empty or 'service' not in df.columns:
            services = ['EC2', 'RDS', 'Lambda', 'S3']
            costs = [150, 80, 45, 30]
        else:
            service_costs = df.groupby('service')['cost'].sum()
            services = service_costs.index.tolist()[:6]  # Limit to 6 services
            costs = service_costs.values.tolist()[:6]
        
        # Create circular layout
        n = len(services)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        x_nodes = np.cos(angles) * 2
        y_nodes = np.sin(angles) * 2
        
        fig = go.Figure()
        
        # Add edges (connections between services)
        for i in range(n):
            for j in range(i+1, min(i+3, n)):  # Connect to next 2 services
                fig.add_trace(go.Scatter(
                    x=[x_nodes[i], x_nodes[j]], 
                    y=[y_nodes[i], y_nodes[j]],
                    mode='lines',
                    line=dict(width=2, color='rgba(100,100,100,0.3)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Add nodes with cost-based sizing
        max_cost = max(costs) if costs else 100
        node_sizes = [20 + (cost/max_cost) * 30 for cost in costs]
        
        fig.add_trace(go.Scatter(
            x=x_nodes, y=y_nodes,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=costs,
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Cost ($)"),
                line=dict(width=2, color='white')
            ),
            text=[f'{s}<br>${c:.0f}' for s, c in zip(services, costs)],
            textposition='middle center',
            textfont=dict(size=10, color='white'),
            name='Services',
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
        
        fig.update_layout(
            title='Service Cost Network',
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            width=600, height=400
        )
        
        return fig
    
    def create_animated_timeseries(self, df: pd.DataFrame) -> go.Figure:
        """Create animated time-series visualization"""
        if df.empty or 'cost' not in df.columns:
            # Create demo data
            dates = pd.date_range('2024-01-01', periods=20, freq='D')
            costs = np.random.uniform(50, 200, 20)
            demo_df = pd.DataFrame({'date': dates, 'cost': costs})
            df = demo_df
        
        # Ensure date column exists
        if 'date' not in df.columns:
            df = df.copy()
            df['date'] = pd.date_range('2024-01-01', periods=len(df), freq='D')
        
        # Create animated line chart
        fig = go.Figure()
        
        # Add main timeline
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['cost'],
            mode='lines+markers',
            name='Cost Timeline',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8, color='#ff7f0e'),
            hovertemplate='Date: %{x}<br>Cost: $%{y:.2f}<extra></extra>'
        ))
        
        # Add trend line
        if len(df) > 1:
            z = np.polyfit(range(len(df)), df['cost'], 1)
            trend_line = np.poly1d(z)(range(len(df)))
            
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='Trend: $%{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Cost Timeline with Trend Analysis',
            xaxis_title='Date',
            yaxis_title='Cost ($)',
            showlegend=True,
            width=800, height=400,
            hovermode='x unified'
        )
        
        return fig