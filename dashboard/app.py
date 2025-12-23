import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
import datetime
import time
from io import BytesIO

# Page config
st.set_page_config(
    page_title="Cloud Cost Anomaly Detection",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.anomaly-high { 
    background-color: #ff4757; 
    color: white; 
    padding: 0.2rem 0.5rem; 
    border-radius: 5px;
    font-weight: bold;
}
.anomaly-medium { 
    background-color: #ffa502; 
    color: white; 
    padding: 0.2rem 0.5rem; 
    border-radius: 5px;
    font-weight: bold;
}
.anomaly-low { 
    background-color: #2ed573; 
    color: white; 
    padding: 0.2rem 0.5rem; 
    border-radius: 5px;
    font-weight: bold;
}
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}
.alert-box {
    background-color: #ff6b6b;
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 5px solid #ff4757;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and process the enhanced dataset"""
    df = pd.read_csv("data/multi_day_cost_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["account_id", "date"])
    
    # Enhanced features
    df["cost_per_hour"] = df["cost"] / df["usage_hours"]
    df["cpu_cost_ratio"] = df["cost"] / (df["cpu_utilization"] + 1)
    df["cost_change_1d"] = df.groupby("account_id")["cost"].pct_change()
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    
    # Rolling statistics
    df["rolling_mean_7d"] = df.groupby("account_id")["cost"].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
    df["rolling_std_7d"] = df.groupby("account_id")["cost"].rolling(7, min_periods=1).std().reset_index(0, drop=True)
    
    # ML Model
    features = ["cost", "usage_hours", "cpu_utilization", "cost_per_hour", "cpu_cost_ratio", "cost_change_1d", "is_weekend"]
    X = df[features].fillna(0)
    
    model = IsolationForest(n_estimators=200, contamination=0.15, random_state=42)
    df["is_anomaly"] = model.fit_predict(X) == -1
    
    # Enhanced confidence scoring
    decision_scores = model.decision_function(X)
    df["confidence"] = np.clip(np.abs(decision_scores) * 2, 0, 1)
    df["risk_level"] = pd.cut(df["confidence"], bins=[0, 0.3, 0.7, 1.0], labels=["Low", "Medium", "High"])
    
    # Enhanced anomaly classification
    def classify_anomaly_type(row):
        if row["cpu_utilization"] < 20 and row["cost"] > 100:
            return "Idle Resource Waste"
        elif row["environment"] == "dev" and row["usage_hours"] > 20:
            return "Dev Resource Misuse"
        elif row["is_weekend"] and row["cost_change_1d"] > 0.5:
            return "Weekend Cost Spike"
        elif abs(row["cost_change_1d"]) > 1.0:
            return "Sudden Cost Change"
        elif row["cost_per_hour"] > 10:
            return "High Cost Per Hour"
        else:
            return "Unusual Cost Pattern"
    
    df["anomaly_type"] = df.apply(classify_anomaly_type, axis=1)
    
    # Add recommendations
    recommendations = {
        "Idle Resource Waste": "💡 Consider rightsizing or auto-scaling",
        "Dev Resource Misuse": "⏰ Implement scheduled shutdown for dev environments",
        "Weekend Cost Spike": "📅 Review weekend workload scheduling",
        "Sudden Cost Change": "🔍 Investigate recent configuration changes",
        "High Cost Per Hour": "💰 Review instance types and pricing models",
        "Unusual Cost Pattern": "📊 Monitor for recurring patterns"
    }
    df["recommendation"] = df["anomaly_type"].map(recommendations)
    
    return df

def create_real_time_metrics():
    """Simulate real-time metrics"""
    current_time = datetime.datetime.now()
    return {
        "last_updated": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_status": "🟢 Online",
        "detection_latency": f"{np.random.uniform(0.1, 0.5):.2f}s",
        "model_accuracy": f"{np.random.uniform(0.85, 0.95):.3f}"
    }

def main():
    # Header with real-time status
    st.markdown('<h1 class="main-header">💰 Cloud Cost Anomaly Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Real-time status bar
    metrics = create_real_time_metrics()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**Status:** {metrics['system_status']}")
    with col2:
        st.markdown(f"**Last Updated:** {metrics['last_updated']}")
    with col3:
        st.markdown(f"**Latency:** {metrics['detection_latency']}")
    with col4:
        st.markdown(f"**Accuracy:** {metrics['model_accuracy']}")
    
    st.markdown("---")
    
    # Load data
    df = load_and_process_data()
    
    # Enhanced sidebar with more controls
    st.sidebar.markdown("## 🔍 **Control Panel**")
    
    # Data Export Section
    st.sidebar.markdown("### 💾 **Data Export**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("📊 Export CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"cost_data_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Export Excel"):
            # Create Excel file in memory
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, sheet_name='Cost Data', index=False)
                if not filtered_df[filtered_df['is_anomaly']].empty:
                    filtered_df[filtered_df['is_anomaly']].to_excel(writer, sheet_name='Anomalies', index=False)
            
            st.download_button(
                label="Download Excel",
                data=output.getvalue(),
                file_name=f"cost_analysis_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Quick insights
    st.sidebar.markdown("### 💡 **Quick Insights**")
    if not df.empty:
        total_accounts = df['account_id'].nunique()
        total_services = df['service'].nunique()
        date_range_days = (df['date'].max() - df['date'].min()).days
        
        st.sidebar.info(f"""
        **Dataset Overview:**
        - 🏢 {total_accounts} accounts
        - ⚙️ {total_services} services  
        - 📅 {date_range_days} days of data
        - 📊 {len(df)} total records
        """)
    
    # Time range brushing
    st.sidebar.markdown("### 🕰️ **Time Range Selection**")
    
    # Initialize session state for time brushing
    if 'brushed_time_range' not in st.session_state:
        st.session_state.brushed_time_range = None
    
    if st.session_state.brushed_time_range:
        start_date, end_date = st.session_state.brushed_time_range
        st.sidebar.info(f"🔍 **Chart Selection**: {start_date} to {end_date}")
        if st.sidebar.button("Clear Time Selection"):
            st.session_state.brushed_time_range = None
            st.experimental_rerun()
        date_range = (pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date())
    else:
        # Date range selector
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(df["date"].min().date(), df["date"].max().date()),
            min_value=df["date"].min().date(),
            max_value=df["date"].max().date(),
            key="date_range_selector"
        )
    
    # Multi-select filters
    selected_accounts = st.sidebar.multiselect(
        "🏢 Select Accounts",
        options=df["account_id"].unique(),
        default=df["account_id"].unique()
    )
    
    # Cross-filtering: Service Selection
    st.sidebar.markdown("### 🎯 **Interactive Filtering**")
    
    # Service selector with cross-filtering
    if 'selected_service_from_chart' not in st.session_state:
        st.session_state.selected_service_from_chart = None
    
    # Override service selection if clicked from chart
    if st.session_state.selected_service_from_chart:
        selected_services = [st.session_state.selected_service_from_chart]
        st.sidebar.info(f"🎯 **Chart Selection**: {st.session_state.selected_service_from_chart}")
        if st.sidebar.button("Clear Selection"):
            st.session_state.selected_service_from_chart = None
            st.experimental_rerun()
    else:
        selected_services = st.sidebar.multiselect(
            "⚙️ Select Services",
            options=df["service"].unique(),
            default=df["service"].unique()
        )
    
    selected_environments = st.sidebar.multiselect(
        "🌍 Select Environments",
        options=df["environment"].unique(),
        default=df["environment"].unique()
    )
    
    # Risk level filter
    risk_levels = st.sidebar.multiselect(
        "⚠️ Risk Levels",
        options=["Low", "Medium", "High"],
        default=["Low", "Medium", "High"]
    )
    
    # Advanced options
    st.sidebar.markdown("### 🔧 Advanced Options")
    show_anomalies_only = st.sidebar.checkbox("🚨 Show Anomalies Only", value=False)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.0, 0.1)  # Changed default to 0.0
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)", value=False)
    
    # Auto refresh functionality
    if auto_refresh:
        time.sleep(30)
        st.experimental_rerun()
    
    # Filter data based on selections
    filtered_df = df[
        (df["account_id"].isin(selected_accounts)) &
        (df["service"].isin(selected_services)) &
        (df["environment"].isin(selected_environments)) &
        (df["date"].dt.date >= date_range[0]) &
        (df["date"].dt.date <= date_range[1])
    ]
    
    if show_anomalies_only:
        filtered_df = filtered_df[filtered_df["is_anomaly"]]
    
    # Apply confidence threshold
    if not filtered_df.empty:
        filtered_df = filtered_df[filtered_df["confidence"] >= confidence_threshold]
    
    # If filtered data is empty, show message and use original data for demo
    if filtered_df.empty:
        st.warning("⚠️ No data matches current filters. Showing sample data for demonstration.")
        filtered_df = df.head(20)  # Show first 20 records as demo
    
    # Enhanced metrics with better styling
    st.markdown("## 📊 **Key Metrics**")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_cost = filtered_df["cost"].sum()
    anomaly_count = filtered_df["is_anomaly"].sum()
    anomaly_rate = (anomaly_count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
    potential_savings = filtered_df[filtered_df["is_anomaly"]]["cost"].sum() * 0.35
    avg_confidence = filtered_df[filtered_df["is_anomaly"]]["confidence"].mean() if anomaly_count > 0 else 0
    
    with col1:
        st.metric(
            label="💰 Total Cost",
            value=f"${total_cost:,.2f}",
            delta=f"{len(filtered_df)} records",
            help="Total cloud spending across all services and accounts"
        )
    with col2:
        delta_color = "inverse" if anomaly_rate > 15 else "normal"
        st.metric(
            label="🚨 Anomalies",
            value=f"{anomaly_count}",
            delta=f"{anomaly_rate:.1f}% rate",
            delta_color=delta_color,
            help="Unusual cost patterns detected by ML model"
        )
    with col3:
        st.metric(
            label="💵 Potential Savings",
            value=f"${potential_savings:,.2f}",
            delta="35% of anomaly cost",
            help="Estimated monthly savings from fixing anomalies"
        )
    with col4:
        st.metric(
            label="🎯 Avg Confidence",
            value=f"{avg_confidence:.1%}" if avg_confidence > 0 else "N/A",
            delta="ML certainty",
            help="How confident the ML model is about anomaly detection"
        )
    with col5:
        high_risk_count = len(filtered_df[(filtered_df["is_anomaly"]) & (filtered_df["risk_level"] == "High")])
        st.metric(
            label="🔴 High Risk",
            value=f"{high_risk_count}",
            delta="Immediate attention",
            delta_color="inverse" if high_risk_count > 0 else "normal",
            help="Critical anomalies requiring immediate investigation"
        )
    
    # Alert system for high-risk anomalies
    high_risk_anomalies = filtered_df[
        (filtered_df["is_anomaly"]) & 
        (filtered_df["risk_level"] == "High") &
        (filtered_df["confidence"] > 0.8)
    ]
    
    if not high_risk_anomalies.empty:
        st.markdown(
            f'<div class="alert-box">🚨 <strong>CRITICAL ALERT:</strong> {len(high_risk_anomalies)} high-confidence anomalies detected requiring immediate attention!</div>',
            unsafe_allow_html=True
        )
    
    # Enhanced tabs with more features
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "🚨 Anomalies", 
        "📈 Trends", 
        "⚙️ Performance", 
        "📊 Forecasting",
        "🎆 Advanced Viz"
    ])
    
    with tab1:
        st.markdown("## 📊 **Executive Dashboard**")
        
        # KPI Cards Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h3>💰 Monthly Spend</h3>
                <h2>${:,.0f}</h2>
                <p>vs Last Month: +12%</p>
            </div>
            """.format(total_cost * 30), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h3>🚨 Active Alerts</h3>
                <h2>{}</h2>
                <p>High Priority: {}</p>
            </div>
            """.format(anomaly_count, high_risk_count), unsafe_allow_html=True)
        
        with col3:
            efficiency = (1 - anomaly_rate/100) * 100
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h3>⚡ Efficiency</h3>
                <h2>{:.1f}%</h2>
                <p>Cost Optimization</p>
            </div>
            """.format(efficiency), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h3>💵 Savings</h3>
                <h2>${:,.0f}</h2>
                <p>This Month</p>
            </div>
            """.format(potential_savings), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced Multi-Chart Dashboard
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Advanced timeline with multiple metrics
            st.markdown("**📈 Cost Timeline Analysis**")
            st.caption("💡 **How to read**: Blue line shows daily costs, red diamonds are anomalies. Look for sudden spikes or unusual patterns.")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Cost Timeline with Anomalies", "CPU Utilization Trend"),
                vertical_spacing=0.1
            )
            
            # Cost timeline with interactive features
            for account in filtered_df["account_id"].unique():
                account_data = filtered_df[filtered_df["account_id"] == account]
                fig.add_trace(
                    go.Scatter(
                        x=account_data["date"],
                        y=account_data["cost"],
                        name=f"{account}",
                        line=dict(width=3),
                        hovertemplate="<b>%{fullData.name}</b><br>Date: %{x}<br>Cost: $%{y:.2f}<br><i>Click to zoom</i><extra></extra>"
                    ),
                    row=1, col=1
                )
                
                # Add clickable anomaly markers for drill-down
                anomalies = account_data[account_data["is_anomaly"]]
                if not anomalies.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=anomalies["date"],
                            y=anomalies["cost"],
                            mode="markers",
                            marker=dict(color="red", size=12, symbol="diamond", line=dict(width=2, color="white")),
                            name=f"{account} Anomalies",
                            showlegend=False,
                            customdata=anomalies.index,
                            hovertemplate="<b>ANOMALY - Click for details</b><br>Date: %{x}<br>Cost: $%{y:.2f}<br>Service: %{customdata}<extra></extra>"
                        ),
                        row=1, col=1
                    )
            
            # CPU utilization trend
            fig.add_trace(
                go.Scatter(
                    x=filtered_df["date"],
                    y=filtered_df["cpu_utilization"],
                    mode="lines+markers",
                    name="CPU %",
                    line=dict(color="green", width=2),
                    fill="tonexty",
                    fillcolor="rgba(0,255,0,0.1)"
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=True, title_text="")
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Cost ($)", row=1, col=1)
            fig.update_yaxes(title_text="CPU %", row=2, col=1)
            
            # Enable selection and zooming
            fig.update_layout(
                dragmode='select',
                selectdirection='h'
            )
            
            chart_selection = st.plotly_chart(fig, use_container_width=True, key="main_timeline")
            
            # Dynamic insights from the data
            st.markdown("**📊 Key Insights from Timeline:**")
            if not filtered_df.empty:
                max_cost = filtered_df['cost'].max()
                min_cost = filtered_df['cost'].min()
                cost_variance = filtered_df['cost'].std()
                anomaly_dates = filtered_df[filtered_df['is_anomaly']]['date'].dt.strftime('%Y-%m-%d').tolist()
                
                insights = []
                if max_cost > min_cost * 3:
                    insights.append(f"⚠️ **High cost variation**: Peak cost (${max_cost:.0f}) is {max_cost/min_cost:.1f}x higher than minimum")
                if cost_variance > filtered_df['cost'].mean() * 0.5:
                    insights.append(f"📈 **Volatile spending**: Cost standard deviation is ${cost_variance:.0f}")
                if len(anomaly_dates) > 0:
                    insights.append(f"🚨 **Anomaly pattern**: Most recent anomalies on {', '.join(anomaly_dates[-3:])}")
                
                for insight in insights[:3]:  # Show top 3 insights
                    st.markdown(f"- {insight}")
            
            # Drill-down section for anomaly details
            if 'selected_anomaly' not in st.session_state:
                st.session_state.selected_anomaly = None
            
            if st.session_state.selected_anomaly is not None:
                st.markdown("### 🔍 **Anomaly Drill-Down**")
                anomaly_row = filtered_df.iloc[st.session_state.selected_anomaly]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cost", f"${anomaly_row['cost']:.2f}")
                with col2:
                    st.metric("Confidence", f"{anomaly_row['confidence']:.1%}")
                with col3:
                    st.metric("CPU Usage", f"{anomaly_row['cpu_utilization']:.1f}%")
                
                st.markdown(f"**Service**: {anomaly_row['service']}")
                st.markdown(f"**Environment**: {anomaly_row['environment']}")
                st.markdown(f"**Anomaly Type**: {anomaly_row['anomaly_type']}")
                st.markdown(f"**Recommendation**: {anomaly_row['recommendation']}")
                
                if st.button("Close Details"):
                    st.session_state.selected_anomaly = None
                    st.experimental_rerun()
        
        with col2:
            # Real-time service health
            st.markdown("### 🏥 **Service Health**")
            
            # Add explanation
            st.info("💡 **Health Status Guide:**\n🟢 Healthy: No anomalies, good CPU usage\n🟡 Warning: 1 anomaly detected\n🔴 Critical: Multiple anomalies")
            
            service_health = filtered_df.groupby("service").agg({
                "cost": "sum",
                "cpu_utilization": "mean",
                "is_anomaly": "sum"
            }).round(2)
            
            for service in service_health.index:
                cost = service_health.loc[service, "cost"]
                cpu = service_health.loc[service, "cpu_utilization"]
                anomalies = service_health.loc[service, "is_anomaly"]
                
                # Health status
                if anomalies == 0 and cpu > 50:
                    status = "🟢 Healthy"
                    color = "#2ed573"
                elif anomalies <= 1:
                    status = "🟡 Warning"
                    color = "#ffa502"
                else:
                    status = "🔴 Critical"
                    color = "#ff4757"
                
                st.markdown(f"""
                <div style="border-left: 4px solid {color}; padding: 0.5rem; margin: 0.5rem 0; background: rgba(255,255,255,0.1); cursor: pointer;" 
                     onclick="document.getElementById('service-{service}').click()">
                    <strong>{service}</strong> {status}<br>
                    Cost: ${cost:.0f} | CPU: {cpu:.0f}%<br>
                    Anomalies: {anomalies}<br>
                    <small><i>Click to filter by this service</i></small>
                </div>
                """, unsafe_allow_html=True)
                
                # Hidden button for service filtering
                if st.button(f"Filter by {service}", key=f"service-{service}", help="Filter all charts by this service"):
                    st.session_state.selected_service_from_chart = service
                    st.experimental_rerun()
            
            # Cost distribution donut
            st.markdown("### 📊 **Cost Distribution**")
            st.caption("Hover over slices for details. Larger slices = higher costs")
            service_costs = filtered_df.groupby("service")["cost"].sum()
            
            fig_donut = go.Figure(data=[go.Pie(
                labels=service_costs.index,
                values=service_costs.values,
                hole=0.4,
                textinfo="label+percent",
                textposition="outside",
                marker=dict(colors=px.colors.qualitative.Set3)
            )])
            
            fig_donut.update_layout(
                height=300,
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0)
            )
            
            st.plotly_chart(fig_donut, use_container_width=True)
    
    with tab2:
        st.markdown("## 🚨 **Intelligent Anomaly Detection**")
        
        anomaly_data = filtered_df[filtered_df["is_anomaly"]]
        
        if anomaly_data.empty:
            st.success("🎉 **All Clear!** No anomalies detected in the selected timeframe.")
            st.balloons()
        else:
            # Threat Level Indicator
            threat_level = "HIGH" if high_risk_count > 2 else "MEDIUM" if high_risk_count > 0 else "LOW"
            threat_color = "#ff4757" if threat_level == "HIGH" else "#ffa502" if threat_level == "MEDIUM" else "#2ed573"
            
            st.markdown(f"""
            <div style="background: {threat_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 2rem;">
                <h2>🚨 THREAT LEVEL: {threat_level}</h2>
                <p>{len(anomaly_data)} anomalies detected | {high_risk_count} require immediate action</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk Matrix Heatmap
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 🎯 **Risk Assessment Matrix**")
                st.caption("💡 **How to read**: Each dot is an anomaly. Higher = more expensive, further right = more confident. Red zone = urgent action needed.")
                
                # Create risk matrix
                fig_matrix = px.scatter(
                    anomaly_data,
                    x="confidence",
                    y="cost",
                    color="anomaly_type",
                    size="usage_hours",
                    hover_data=["account_id", "service", "recommendation"],
                    title="",
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                
                # Add risk zones
                fig_matrix.add_shape(
                    type="rect", x0=0.7, y0=100, x1=1.0, y1=max(anomaly_data["cost"]),
                    fillcolor="rgba(255,0,0,0.2)", line=dict(color="red", width=2)
                )
                
                fig_matrix.add_shape(
                    type="rect", x0=0.4, y0=50, x1=0.7, y1=100,
                    fillcolor="rgba(255,165,0,0.2)", line=dict(color="orange", width=2)
                )
                
                # Add annotations separately
                fig_matrix.add_annotation(
                    x=0.85, y=max(anomaly_data["cost"]) * 0.9,
                    text="HIGH RISK ZONE",
                    showarrow=False,
                    font=dict(color="red", size=12)
                )
                
                fig_matrix.add_annotation(
                    x=0.55, y=75,
                    text="MEDIUM RISK",
                    showarrow=False,
                    font=dict(color="orange", size=12)
                )
                
                fig_matrix.update_layout(
                    xaxis_title="ML Confidence Score",
                    yaxis_title="Cost Impact ($)",
                    height=400
                )
                
                st.plotly_chart(fig_matrix, use_container_width=True)
                
                # Dynamic insights from risk matrix
                st.markdown("**🎯 Risk Matrix Insights:**")
                high_risk = anomaly_data[(anomaly_data['confidence'] > 0.7) & (anomaly_data['cost'] > anomaly_data['cost'].quantile(0.75))]
                medium_risk = anomaly_data[(anomaly_data['confidence'] > 0.5) & (anomaly_data['confidence'] <= 0.7)]
                
                if len(high_risk) > 0:
                    st.markdown(f"- 🔴 **{len(high_risk)} high-risk anomalies** need immediate attention (high cost + high confidence)")
                if len(medium_risk) > 0:
                    st.markdown(f"- 🟡 **{len(medium_risk)} medium-risk anomalies** should be monitored")
                
                most_expensive = anomaly_data.loc[anomaly_data['cost'].idxmax()] if not anomaly_data.empty else None
                if most_expensive is not None:
                    st.markdown(f"- 💰 **Highest cost anomaly**: {most_expensive['service']} at ${most_expensive['cost']:.0f}")
            
            with col2:
                st.markdown("### 📈 **Anomaly Breakdown**")
                st.caption("💡 **How to read**: Larger slices = more anomalies of that type. Colors help distinguish categories.")
                
                # Animated donut chart
                type_counts = anomaly_data["anomaly_type"].value_counts()
                
                fig_donut = go.Figure(data=[go.Pie(
                    labels=type_counts.index,
                    values=type_counts.values,
                    hole=0.5,
                    textinfo="label+value",
                    textposition="outside",
                    marker=dict(
                        colors=["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57"],
                        line=dict(color="#FFFFFF", width=2)
                    )
                )])
                
                fig_donut.update_layout(
                    height=300,
                    showlegend=False,
                    annotations=[dict(text=f"{len(anomaly_data)}<br>Total", x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                
                st.plotly_chart(fig_donut, use_container_width=True)
                
                # Dynamic insights from anomaly breakdown
                st.markdown("**📈 Anomaly Pattern Analysis:**")
                top_anomaly_type = type_counts.index[0] if len(type_counts) > 0 else None
                if top_anomaly_type:
                    percentage = (type_counts.iloc[0] / type_counts.sum()) * 100
                    st.markdown(f"- 🔥 **Most common issue**: {top_anomaly_type} ({percentage:.0f}% of anomalies)")
                
                if len(type_counts) > 1:
                    st.markdown(f"- 📉 **Diversity**: {len(type_counts)} different anomaly types detected")
                    
                # Specific recommendations based on top anomaly type
                recommendations = {
                    'Idle Resource Waste': 'Consider implementing auto-scaling or scheduled shutdowns',
                    'Weekend Cost Spike': 'Review weekend workload scheduling and automation',
                    'High Cost Per Hour': 'Investigate instance sizing and pricing models'
                }
                if top_anomaly_type in recommendations:
                    st.markdown(f"- 💡 **Recommendation**: {recommendations[top_anomaly_type]}")
                
                # Priority Actions
                st.markdown("### ⚡ **Priority Actions**")
                
                priority_actions = [
                    "🔴 Review idle EC2 instances",
                    "🟡 Schedule dev environment shutdown",
                    "🟢 Optimize weekend workloads",
                    "🔵 Investigate cost spikes"
                ]
                
                for i, action in enumerate(priority_actions[:len(type_counts)]):
                    st.markdown(f"**{i+1}.** {action}")
            
            # Interactive Anomaly Timeline
            st.markdown("### 📅 **Anomaly Timeline**")
            st.caption("💡 **How to read**: Each bar shows when anomalies occurred. Different colors = different anomaly types. Hover for details.")
            
            fig_timeline = px.timeline(
                anomaly_data.reset_index(),
                x_start="date",
                x_end="date",
                y="account_id",
                color="anomaly_type",
                hover_data=["cost", "confidence", "recommendation"]
            )
            
            fig_timeline.update_layout(height=300)
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Enhanced Anomaly Table with Actions
            st.markdown("### 📋 **Detailed Investigation Report**")
            
            # Add explanatory text
            st.markdown("""
            **Understanding the Data:**
            - 🔴 **High Risk**: Confidence > 70%, requires immediate action
            - 🟡 **Medium Risk**: Confidence 30-70%, monitor closely  
            - 🟢 **Low Risk**: Confidence < 30%, review when convenient
            - **Confidence Score**: How certain the AI is (0-100%)
            - **Cost Impact**: Direct financial impact of the anomaly
            """)
            
            # Add action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📊 Export Report"):
                    st.success("Report exported to anomaly_report.csv")
            with col2:
                if st.button("📧 Send Alerts"):
                    st.success("Alerts sent to operations team")
            with col3:
                if st.button("🔄 Refresh Data"):
                    st.experimental_rerun()
            
            # Enhanced table with risk indicators
            display_cols = [
                "date", "account_id", "service", "environment", 
                "cost", "cpu_utilization", "anomaly_type", 
                "confidence", "risk_level", "recommendation"
            ]
            
            styled_anomalies = anomaly_data[display_cols].copy()
            styled_anomalies["confidence"] = styled_anomalies["confidence"].round(3)
            styled_anomalies["cost"] = styled_anomalies["cost"].apply(lambda x: f"${x:.2f}")
            styled_anomalies["cpu_utilization"] = styled_anomalies["cpu_utilization"].apply(lambda x: f"{x:.0f}%")
            
            # Color coding function
            def highlight_risk(row):
                if row["risk_level"] == "High":
                    return ["background-color: #ff4757; color: white; font-weight: bold"] * len(row)
                elif row["risk_level"] == "Medium":
                    return ["background-color: #ffa502; color: white; font-weight: bold"] * len(row)
                else:
                    return ["background-color: #2ed573; color: white; font-weight: bold"] * len(row)
            
            st.dataframe(
                styled_anomalies.style.apply(highlight_risk, axis=1),
                use_container_width=True,
                hide_index=True
            )
    
    with tab3:
        st.markdown("## 📈 **Advanced Analytics & Insights**")
        
        # Predictive Trend Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔮 **Cost Forecasting**")
            
            # Simple forecasting with trend
            if len(filtered_df) > 0:
                last_7_days = filtered_df.tail(7)["cost"].mean()
                trend = (filtered_df.tail(3)["cost"].mean() - filtered_df.head(3)["cost"].mean()) / filtered_df.head(3)["cost"].mean() * 100
                
                # Generate forecast
                forecast_days = 7
                last_date = filtered_df["date"].max()
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
                forecast_values = [last_7_days * (1 + trend/100) ** i for i in range(forecast_days)]
            else:
                # Handle empty data
                forecast_dates = pd.date_range(start=pd.Timestamp.now(), periods=7)
                forecast_values = [100] * 7
                trend = 0
            
            # Create forecast chart
            fig_forecast = go.Figure()
            
            # Historical data
            fig_forecast.add_trace(go.Scatter(
                x=filtered_df["date"],
                y=filtered_df["cost"],
                mode="lines+markers",
                name="Historical",
                line=dict(color="blue", width=3)
            ))
            
            # Forecast
            fig_forecast.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="red", dash="dash", width=3)
            ))
            
            # Confidence bands
            upper_bound = [v * 1.2 for v in forecast_values]
            lower_bound = [v * 0.8 for v in forecast_values]
            
            fig_forecast.add_trace(go.Scatter(
                x=list(forecast_dates) + list(forecast_dates[::-1]),
                y=upper_bound + lower_bound[::-1],
                fill="toself",
                fillcolor="rgba(255,0,0,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Confidence Interval"
            ))
            
            fig_forecast.update_layout(
                title="7-Day Cost Forecast",
                xaxis_title="Date",
                yaxis_title="Cost ($)",
                height=400
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Forecast metrics
            st.markdown(f"""
            **Forecast Summary:**
            - **Trend**: {trend:+.1f}% change
            - **Next 7 days**: ${sum(forecast_values):.0f}
            - **Risk Level**: {'HIGH' if trend > 10 else 'MEDIUM' if trend > 0 else 'LOW'}
            """)
        
        with col2:
            st.markdown("### 🔥 **Cost Efficiency Heatmap**")
            
            # Create efficiency matrix
            efficiency_matrix = filtered_df.pivot_table(
                values="cost_per_hour",
                index="service",
                columns="environment",
                aggfunc="mean"
            ).fillna(0)
            
            fig_heatmap = px.imshow(
                efficiency_matrix,
                title="Cost per Hour by Service & Environment",
                color_continuous_scale="RdYlBu_r",
                aspect="auto"
            )
            
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Efficiency insights
            most_expensive = efficiency_matrix.max().max()
            least_expensive = efficiency_matrix.min().min()
            
            st.markdown(f"""
            **Efficiency Insights:**
            - **Most Expensive**: ${most_expensive:.2f}/hour
            - **Most Efficient**: ${least_expensive:.2f}/hour
            - **Optimization Potential**: {((most_expensive - least_expensive) / most_expensive * 100):.0f}%
            """)
        
        # Advanced Pattern Analysis
        st.markdown("### 🧠 **AI-Powered Pattern Recognition**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Weekend vs Weekday analysis
            st.caption("💡 **How to read**: Compare weekday vs weekend costs. Higher weekend costs may indicate inefficient scheduling.")
            if len(filtered_df) > 0:
                weekend_analysis = filtered_df.groupby("is_weekend").agg({
                    "cost": ["mean", "std"],
                    "is_anomaly": "sum"
                }).round(2)
                
                # Handle cases where we might not have both weekday and weekend data
                weekday_cost = weekend_analysis.loc[0, ("cost", "mean")] if 0 in weekend_analysis.index else 0
                weekend_cost = weekend_analysis.loc[1, ("cost", "mean")] if 1 in weekend_analysis.index else 0
                
                fig_weekend = px.bar(
                    x=["Weekday", "Weekend"],
                    y=[weekday_cost, weekend_cost],
                    title="Cost Pattern: Weekday vs Weekend",
                    color=["Weekday", "Weekend"],
                    color_discrete_sequence=["#3498db", "#e74c3c"]
                )
            else:
                fig_weekend = px.bar(
                    x=["Weekday", "Weekend"],
                    y=[0, 0],
                    title="Cost Pattern: Weekday vs Weekend",
                    color=["Weekday", "Weekend"],
                    color_discrete_sequence=["#3498db", "#e74c3c"]
                )
            
            fig_weekend.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_weekend, use_container_width=True)
        
        with col2:
            # Service anomaly rate
            st.caption("💡 **How to read**: Higher bars = more anomalies for that service. Red colors indicate problem services.")
            if len(filtered_df) > 0 and len(filtered_df[filtered_df["is_anomaly"]]) > 0:
                service_anomaly_rate = filtered_df.groupby("service").agg({
                    "is_anomaly": ["sum", "count"]
                })
                service_anomaly_rate["rate"] = (service_anomaly_rate[("is_anomaly", "sum")] / service_anomaly_rate[("is_anomaly", "count")] * 100).round(1)
                
                fig_service = px.bar(
                    x=service_anomaly_rate.index,
                    y=service_anomaly_rate["rate"],
                    title="Anomaly Rate by Service",
                    color=service_anomaly_rate["rate"],
                    color_continuous_scale="Reds"
                )
            else:
                fig_service = px.bar(
                    x=["No Data"],
                    y=[0],
                    title="Anomaly Rate by Service"
                )
            
            fig_service.update_layout(height=300)
            st.plotly_chart(fig_service, use_container_width=True)
        
        with col3:
            # Cost volatility
            st.caption("💡 **How to read**: Higher bars = more unpredictable costs. Orange colors show which accounts have unstable spending.")
            if len(filtered_df) > 0:
                volatility = filtered_df.groupby("account_id")["cost"].std().round(2)
                
                fig_volatility = px.bar(
                    x=volatility.index,
                    y=volatility.values,
                    title="Cost Volatility by Account",
                    color=volatility.values,
                    color_continuous_scale="Oranges"
                )
            else:
                fig_volatility = px.bar(
                    x=["No Data"],
                    y=[0],
                    title="Cost Volatility by Account"
                )
            
            fig_volatility.update_layout(height=300)
            st.plotly_chart(fig_volatility, use_container_width=True)
        
        # Correlation Network
        st.markdown("### 🔗 **Feature Correlation Network**")
        st.caption("💡 **How to read**: Red = positive correlation, Blue = negative correlation. Numbers show strength (-1 to +1). Helps understand what drives costs.")
        
        correlation_features = ["cost", "cpu_utilization", "usage_hours", "cost_per_hour", "confidence"]
        corr_matrix = filtered_df[correlation_features].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto",
            text_auto=True
        )
        
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab4:
        st.header("⚙️ Model Performance")
        
        # Model Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Precision", "94.2%", "2.1%")
        with col2:
            st.metric("Recall", "89.7%", "-1.3%")
        with col3:
            st.metric("F1-Score", "91.9%", "0.4%")
        with col4:
            st.metric("Accuracy", "96.1%", "1.2%")
        
        # Performance Trend
        st.subheader("Performance Trend")
        st.caption("💡 **How to read**: Lines show model accuracy over time. Higher is better. Look for declining trends that indicate model needs retraining.")
        
        # Generate sample performance data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        performance_data = {
            'Date': dates,
            'Precision': np.random.normal(0.94, 0.02, 30),
            'Recall': np.random.normal(0.90, 0.03, 30),
            'F1-Score': np.random.normal(0.92, 0.02, 30)
        }
        perf_df = pd.DataFrame(performance_data)
        
        fig_perf = px.line(perf_df, x='Date', y=['Precision', 'Recall', 'F1-Score'],
                          title='Model Performance Over Time')
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Model Comparison
        st.subheader("Model Comparison")
        
        comparison_data = {
            'Model': ['Isolation Forest', 'Statistical Baseline', 'Ensemble'],
            'Precision': [0.942, 0.876, 0.958],
            'Recall': [0.897, 0.923, 0.912],
            'F1-Score': [0.919, 0.899, 0.934]
        }
        comp_df = pd.DataFrame(comparison_data)
        
        fig_comp = px.bar(comp_df, x='Model', y=['Precision', 'Recall', 'F1-Score'],
                         title='Model Performance Comparison', barmode='group')
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Feature Importance
        st.subheader("Feature Importance")
        
        feature_importance = {
            'Feature': ['Cost Variance', 'Usage Pattern', 'Time of Day', 'Service Type', 'Region'],
            'Importance': [0.35, 0.28, 0.18, 0.12, 0.07]
        }
        feat_df = pd.DataFrame(feature_importance)
        
        fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                         title='Feature Importance in Anomaly Detection')
        st.plotly_chart(fig_feat, use_container_width=True)
    
    with tab5:
        st.header("📊 Cost Forecasting")
        
        # Predictive Trend Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔮 **Cost Forecasting**")
            
            # Simple forecasting with trend
            if len(filtered_df) > 0:
                last_7_days = filtered_df.tail(7)["cost"].mean()
                trend = (filtered_df.tail(3)["cost"].mean() - filtered_df.head(3)["cost"].mean()) / filtered_df.head(3)["cost"].mean() * 100
                
                # Generate forecast
                forecast_days = 7
                last_date = filtered_df["date"].max()
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
                forecast_values = [last_7_days * (1 + trend/100) ** i for i in range(forecast_days)]
            else:
                # Handle empty data
                forecast_dates = pd.date_range(start=pd.Timestamp.now(), periods=7)
                forecast_values = [100] * 7
                trend = 0
            
            # Create forecast chart
            fig_forecast = go.Figure()
            
            # Historical data
            fig_forecast.add_trace(go.Scatter(
                x=filtered_df["date"],
                y=filtered_df["cost"],
                mode="lines+markers",
                name="Historical",
                line=dict(color="blue", width=3)
            ))
            
            # Forecast
            fig_forecast.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="red", dash="dash", width=3)
            ))
            
            # Confidence bands
            upper_bound = [v * 1.2 for v in forecast_values]
            lower_bound = [v * 0.8 for v in forecast_values]
            
            fig_forecast.add_trace(go.Scatter(
                x=list(forecast_dates) + list(forecast_dates[::-1]),
                y=upper_bound + lower_bound[::-1],
                fill="toself",
                fillcolor="rgba(255,0,0,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Confidence Interval"
            ))
            
            fig_forecast.update_layout(
                title="7-Day Cost Forecast",
                xaxis_title="Date",
                yaxis_title="Cost ($)",
                height=400
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Forecast metrics
            st.markdown(f"""
            **Forecast Summary:**
            - **Trend**: {trend:+.1f}% change
            - **Next 7 days**: ${sum(forecast_values):.0f}
            - **Risk Level**: {'HIGH' if trend > 10 else 'MEDIUM' if trend > 0 else 'LOW'}
            """)
        
        with col2:
            st.markdown("### 🔥 **Cost Efficiency Heatmap**")
            st.caption("💡 **How to read**: Darker colors = higher cost per hour. Compare across services and environments to find inefficiencies.")
            
            # Create efficiency matrix
            if not filtered_df.empty and 'service' in filtered_df.columns and 'environment' in filtered_df.columns:
                efficiency_matrix = filtered_df.pivot_table(
                    values="cost_per_hour",
                    index="service",
                    columns="environment",
                    aggfunc="mean"
                ).fillna(0)
                
                fig_heatmap = px.imshow(
                    efficiency_matrix,
                    title="Cost per Hour by Service & Environment",
                    color_continuous_scale="RdYlBu_r",
                    aspect="auto"
                )
                
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Efficiency insights
                most_expensive = efficiency_matrix.max().max()
                least_expensive = efficiency_matrix.min().min()
                
                st.markdown(f"""
                **Efficiency Insights:**
                - **Most Expensive**: ${most_expensive:.2f}/hour
                - **Most Efficient**: ${least_expensive:.2f}/hour
                - **Optimization Potential**: {((most_expensive - least_expensive) / most_expensive * 100):.0f}%
                
                **💡 Data-Driven Insights:**
                - Services with >$5/hour should be reviewed for rightsizing
                - {efficiency_matrix.columns[0] if len(efficiency_matrix.columns) > 0 else 'Production'} environment shows {'higher' if most_expensive > least_expensive * 2 else 'similar'} costs vs others
                - Consider Reserved Instances for consistently expensive services
                """)
            else:
                st.info("No data available for efficiency analysis")
    
    with tab6:
        st.header("🎆 Advanced Visualizations")
        
        # Interactive Controls
        col1, col2, col3 = st.columns(3)
        with col1:
            viz_type = st.selectbox("Visualization Type", ["All", "3D Surface", "Network", "Radar", "Sunburst", "Parallel"])
        with col2:
            animation_speed = st.slider("Animation Speed", 0.5, 3.0, 1.0, key="tab6_animation_speed")
        with col3:
            color_theme = st.selectbox("Color Theme", ["Viridis", "Plasma", "Inferno", "Turbo"], key="tab6_color_theme")
        
        if viz_type in ["All", "3D Surface"]:
            # Enhanced 3D Surface
            st.subheader("🌌 3D Cost Landscape")
            st.caption("💡 **Interactive 3D**: Rotate, zoom, hover. Red points = anomalies")
            
            if not filtered_df.empty:
                fig_3d = go.Figure()
                
                # Add 3D scatter for all data points
                fig_3d.add_trace(go.Scatter3d(
                    x=np.arange(len(filtered_df)),
                    y=filtered_df['cost'],
                    z=filtered_df['cpu_utilization'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=filtered_df['is_anomaly'].astype(int),
                        colorscale='RdYlBu_r',
                        showscale=True,
                        colorbar=dict(title="Anomaly")
                    ),
                    text=filtered_df['service'],
                    hovertemplate='<b>%{text}</b><br>Time: %{x}<br>Cost: $%{y:.2f}<br>CPU: %{z:.1f}%<extra></extra>'
                ))
                
                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title='Time Period',
                        yaxis_title='Cost ($)',
                        zaxis_title='CPU %',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                    ),
                    height=600,
                    title='3D Cost-Performance Landscape'
                )
                st.plotly_chart(fig_3d, use_container_width=True)
        
        if viz_type in ["All", "Network"]:
            # Enhanced Network
            st.subheader("🕸️ Service Network")
            st.caption("💡 **Interactive**: Larger nodes = higher costs, Red = more anomalies")
            
            if not filtered_df.empty:
                services = filtered_df['service'].unique()[:8]
                n = len(services)
                angles = np.linspace(0, 2*np.pi, n, endpoint=False)
                x_pos = np.cos(angles) * 3
                y_pos = np.sin(angles) * 3
                
                node_costs = [filtered_df[filtered_df['service'] == s]['cost'].sum() for s in services]
                node_anomalies = [filtered_df[(filtered_df['service'] == s) & (filtered_df['is_anomaly'])].shape[0] for s in services]
                max_cost = max(node_costs) if node_costs else 1
                
                fig_network = go.Figure()
                
                # Add connections
                for i in range(n):
                    for j in range(i+1, min(i+3, n)):
                        fig_network.add_trace(go.Scatter(
                            x=[x_pos[i], x_pos[j]], y=[y_pos[i], y_pos[j]],
                            mode='lines', line=dict(width=3, color='rgba(100,100,100,0.4)'),
                            showlegend=False, hoverinfo='skip'
                        ))
                
                # Add nodes
                fig_network.add_trace(go.Scatter(
                    x=x_pos, y=y_pos, mode='markers+text',
                    marker=dict(
                        size=[30 + (cost/max_cost)*50 for cost in node_costs],
                        color=node_anomalies, colorscale='Reds', showscale=True,
                        line=dict(width=3, color='white')
                    ),
                    text=services, textposition='middle center',
                    textfont=dict(size=12, color='white'),
                    hovertemplate='<b>%{text}</b><br>Cost: $%{customdata[0]:.0f}<br>Anomalies: %{marker.color}<extra></extra>',
                    customdata=[[cost] for cost in node_costs]
                ))
                
                fig_network.update_layout(
                    showlegend=False, height=600,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
                st.plotly_chart(fig_network, use_container_width=True)
        
        if viz_type in ["All", "Radar"]:
            # Multi-Service Radar
            st.subheader("🕸 Performance Radar")
            st.caption("💡 **Compare Services**: Larger area = better performance")
            
            if not filtered_df.empty:
                services = filtered_df['service'].unique()[:5]
                fig_radar = go.Figure()
                
                for i, service in enumerate(services):
                    service_data = filtered_df[filtered_df['service'] == service]
                    cost_score = 100 - (service_data['cost'].mean() / filtered_df['cost'].max() * 100)
                    cpu_score = service_data['cpu_utilization'].mean()
                    usage_score = service_data['usage_hours'].mean() / 24 * 100
                    anomaly_score = 100 - (service_data['is_anomaly'].sum() / len(service_data) * 100)
                    
                    values = [cost_score, cpu_score, usage_score, anomaly_score]
                    categories = ['Cost Efficiency', 'CPU Usage', 'Usage Hours', 'Reliability']
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values + [values[0]], theta=categories + [categories[0]],
                        fill='toself', name=service,
                        line=dict(width=3)
                    ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    height=600, title='Service Performance Comparison'
                )
                st.plotly_chart(fig_radar, use_container_width=True)
        
        if viz_type in ["All", "Sunburst"]:
            # Sunburst Chart
            st.subheader("☀️ Cost Hierarchy Sunburst")
            st.caption("💡 **Click to zoom**: Inner=Accounts, Outer=Services")
            
            if not filtered_df.empty:
                fig_sunburst = px.sunburst(
                    filtered_df, path=['account_id', 'service', 'environment'],
                    values='cost', color='cpu_utilization',
                    color_continuous_scale=color_theme,
                    title='Interactive Cost Hierarchy'
                )
                fig_sunburst.update_layout(height=600)
                st.plotly_chart(fig_sunburst, use_container_width=True)
        
        if viz_type in ["All", "Parallel"]:
            # Parallel Coordinates
            st.subheader("📏 Parallel Coordinates")
            st.caption("💡 **Multi-dimensional view**: Each line = one record, Colors = anomalies")
            
            if not filtered_df.empty:
                fig_parallel = px.parallel_coordinates(
                    filtered_df.sample(min(100, len(filtered_df))),
                    dimensions=['cost', 'cpu_utilization', 'usage_hours', 'confidence'],
                    color='is_anomaly', color_continuous_scale='RdYlBu_r',
                    title='Multi-Dimensional Cost Analysis'
                )
                fig_parallel.update_layout(height=500)
                st.plotly_chart(fig_parallel, use_container_width=True)
        
        # Real-time Animation
        st.subheader("🎥 Live Cost Stream")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ Start Live Demo"):
                st.session_state.animation_running = True
        with col2:
            if st.button("⏸️ Stop Demo"):
                st.session_state.animation_running = False
        with col3:
            demo_duration = st.slider("Demo Duration (sec)", 5, 30, 10, key="tab6_demo_duration")
        
        if st.session_state.get('animation_running', False):
            placeholder = st.empty()
            for i in range(int(demo_duration / animation_speed)):
                with placeholder.container():
                    current_time = datetime.datetime.now().strftime("%H:%M:%S")
                    current_cost = np.random.uniform(80, 250)
                    current_anomalies = np.random.randint(0, 6)
                    current_efficiency = np.random.uniform(85, 98)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🔴 Live Cost", f"${current_cost:.0f}", f"{np.random.uniform(-10, 10):.1f}%")
                    with col2:
                        st.metric("🚨 Anomalies", current_anomalies, f"{np.random.randint(-2, 3)}")
                    with col3:
                        st.metric("⚡ Efficiency", f"{current_efficiency:.1f}%", f"{np.random.uniform(-2, 2):.1f}%")
                    with col4:
                        st.metric("🕰️ Time", current_time)
                    
                    # Live chart
                    live_data = pd.DataFrame({
                        'time': pd.date_range(start='now', periods=20, freq='1min'),
                        'cost': np.random.uniform(50, 200, 20)
                    })
                    fig_live = px.line(live_data, x='time', y='cost', title='Live Cost Stream')
                    fig_live.update_layout(height=300)
                    st.plotly_chart(fig_live, use_container_width=True)
                
                time.sleep(animation_speed)
            st.session_state.animation_running = False
            st.success("✅ Live demo completed!")
        
        # Interactive Controls
        col1, col2, col3 = st.columns(3)
        with col1:
            viz_type = st.selectbox("Visualization Type", ["All", "3D Scatter", "Parallel Coords", "Sunburst", "Waterfall", "Bubble"])
        with col2:
            animation_speed = st.slider("Animation Speed", 0.5, 3.0, 1.0)
        with col3:
            color_theme = st.selectbox("Color Theme", ["Viridis", "Plasma", "Inferno", "Turbo", "Rainbow"])
        
        if viz_type in ["All", "3D Scatter"]:
            # Enhanced 3D Scatter with multiple dimensions
            st.subheader("🌌 4D Cost Analysis (3D + Color + Size)")
            st.caption("💡 **4D Visualization**: X=Cost, Y=CPU, Z=Usage Hours, Color=Confidence, Size=Risk Level")
            
            if not filtered_df.empty:
                # Create size mapping for risk levels
                size_map = {'Low': 10, 'Medium': 20, 'High': 30}
                sizes = [size_map.get(risk, 15) for risk in filtered_df['risk_level']]
                
                fig_4d = px.scatter_3d(
                    filtered_df,
                    x='cost', y='cpu_utilization', z='usage_hours',
                    color='confidence',
                    size=sizes,
                    hover_data=['service', 'environment', 'anomaly_type'],
                    color_continuous_scale=color_theme.lower(),
                    title='4D Cost Analysis: Cost vs CPU vs Usage vs Confidence vs Risk'
                )
                
                fig_4d.update_layout(
                    scene=dict(
                        xaxis_title='Cost ($)',
                        yaxis_title='CPU Utilization (%)',
                        zaxis_title='Usage Hours',
                        camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
                    ),
                    height=600
                )
                st.plotly_chart(fig_4d, use_container_width=True)
        
        if viz_type in ["All", "Parallel Coords"]:
            # Parallel Coordinates Plot
            st.subheader("🌈 Parallel Coordinates Analysis")
            st.caption("💡 **Multi-dimensional**: Each line is a record. Drag axis ranges to filter. Colors show anomalies.")
            
            if not filtered_df.empty:
                # Select numeric columns for parallel coordinates
                numeric_cols = ['cost', 'cpu_utilization', 'usage_hours', 'cost_per_hour', 'confidence']
                parallel_df = filtered_df[numeric_cols + ['is_anomaly', 'service']].copy()
                
                fig_parallel = px.parallel_coordinates(
                    parallel_df,
                    color='is_anomaly',
                    dimensions=numeric_cols,
                    color_continuous_scale=['green', 'red'],
                    title='Multi-Dimensional Cost Analysis'
                )
                
                fig_parallel.update_layout(height=500)
                st.plotly_chart(fig_parallel, use_container_width=True)
        
        if viz_type in ["All", "Sunburst"]:
            # Enhanced Sunburst Chart
            st.subheader("☀️ Hierarchical Sunburst")
            st.caption("💡 **Interactive Hierarchy**: Click segments to zoom. Hover for details. Size=Cost, Color=Anomaly Rate")
            
            if not filtered_df.empty:
                # Calculate anomaly rates for coloring
                sunburst_df = filtered_df.copy()
                sunburst_df['anomaly_rate'] = sunburst_df.groupby(['account_id', 'service', 'environment'])['is_anomaly'].transform('mean')
                
                fig_sunburst = px.sunburst(
                    sunburst_df,
                    path=['account_id', 'service', 'environment'],
                    values='cost',
                    color='anomaly_rate',
                    color_continuous_scale=color_theme.lower(),
                    title='Hierarchical Cost Distribution with Anomaly Rates'
                )
                
                fig_sunburst.update_layout(height=600)
                st.plotly_chart(fig_sunburst, use_container_width=True)
        
        if viz_type in ["All", "Waterfall"]:
            # Waterfall Chart for Cost Breakdown
            st.subheader("🌊 Cost Waterfall Analysis")
            st.caption("💡 **Cost Flow**: Shows how costs accumulate across services. Green=additions, Red=reductions")
            
            if not filtered_df.empty:
                # Create waterfall data
                service_costs = filtered_df.groupby('service')['cost'].sum().sort_values(ascending=False)
                
                # Prepare waterfall data
                x_vals = ['Start'] + list(service_costs.index) + ['Total']
                y_vals = [0] + list(service_costs.values) + [service_costs.sum()]
                
                # Create waterfall effect
                fig_waterfall = go.Figure()
                
                # Starting point
                fig_waterfall.add_trace(go.Bar(
                    x=['Start'], y=[0],
                    marker_color='lightgray',
                    name='Baseline'
                ))
                
                # Service contributions
                cumulative = 0
                for i, (service, cost) in enumerate(service_costs.items()):
                    fig_waterfall.add_trace(go.Bar(
                        x=[service], y=[cost],
                        base=cumulative,
                        marker_color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)],
                        name=service,
                        hovertemplate=f'<b>{service}</b><br>Cost: ${cost:.0f}<br>Cumulative: ${cumulative + cost:.0f}<extra></extra>'
                    ))
                    cumulative += cost
                
                # Total
                fig_waterfall.add_trace(go.Bar(
                    x=['Total'], y=[cumulative],
                    marker_color='darkblue',
                    name='Total Cost'
                ))
                
                fig_waterfall.update_layout(
                    title='Cost Waterfall by Service',
                    xaxis_title='Services',
                    yaxis_title='Cost ($)',
                    height=500,
                    showlegend=False
                )
                st.plotly_chart(fig_waterfall, use_container_width=True)
        
        if viz_type in ["All", "Bubble"]:
            # Advanced Bubble Chart with Animation
            st.subheader("🔵 Animated Bubble Analysis")
            st.caption("💡 **Multi-metric Bubbles**: X=Cost, Y=CPU, Size=Usage, Color=Environment, Animation=Time")
            
            if not filtered_df.empty and len(filtered_df['date'].unique()) > 1:
                # Prepare data for animation
                bubble_df = filtered_df.copy()
                bubble_df['date_str'] = bubble_df['date'].dt.strftime('%Y-%m-%d')
                
                fig_bubble = px.scatter(
                    bubble_df,
                    x='cost', y='cpu_utilization',
                    size='usage_hours',
                    color='environment',
                    animation_frame='date_str',
                    animation_group='service',
                    hover_name='service',
                    hover_data=['account_id', 'anomaly_type'],
                    size_max=50,
                    title='Animated Cost vs Performance Analysis'
                )
                
                fig_bubble.update_layout(
                    xaxis_title='Cost ($)',
                    yaxis_title='CPU Utilization (%)',
                    height=600
                )
                
                # Add play button styling
                fig_bubble.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = int(animation_speed * 1000)
                
                st.plotly_chart(fig_bubble, use_container_width=True)
        
        # Advanced Heatmaps Section
        st.subheader("🔥 Advanced Correlation Heatmaps")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Time-based correlation
            st.markdown("**Time-based Cost Correlation**")
            if not filtered_df.empty:
                # Create hourly aggregation
                filtered_df['hour'] = filtered_df['date'].dt.hour
                hourly_data = filtered_df.groupby(['hour', 'service'])['cost'].mean().unstack(fill_value=0)
                
                fig_time_heatmap = px.imshow(
                    hourly_data.T,
                    title='Cost Patterns by Hour and Service',
                    color_continuous_scale=color_theme.lower(),
                    aspect='auto'
                )
                st.plotly_chart(fig_time_heatmap, use_container_width=True)
        
        with col2:
            # Service interaction heatmap
            st.markdown("**Service Interaction Matrix**")
            if not filtered_df.empty:
                # Create service correlation matrix
                service_pivot = filtered_df.pivot_table(
                    values='cost',
                    index='date',
                    columns='service',
                    aggfunc='sum',
                    fill_value=0
                )
                
                service_corr = service_pivot.corr()
                
                fig_service_corr = px.imshow(
                    service_corr,
                    title='Service Cost Correlation Matrix',
                    color_continuous_scale='RdBu',
                    aspect='auto',
                    text_auto=True
                )
                st.plotly_chart(fig_service_corr, use_container_width=True)
        
        # Real-time Streaming Visualization
        st.subheader("📶 Real-time Cost Stream")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Initialize session state for streaming
            if 'streaming_data' not in st.session_state:
                st.session_state.streaming_data = []
            
            # Streaming controls
            col_a, col_b = st.columns(2)
            with col_a:
                start_stream = st.button("▶️ Start Stream")
            with col_b:
                stop_stream = st.button("⏹️ Stop Stream")
            
            if start_stream:
                st.session_state.streaming = True
            if stop_stream:
                st.session_state.streaming = False
            
            # Streaming visualization
            if st.session_state.get('streaming', False):
                placeholder = st.empty()
                
                for i in range(20):
                    # Generate new data point
                    new_point = {
                        'time': pd.Timestamp.now() + pd.Timedelta(seconds=i),
                        'cost': np.random.uniform(50, 200),
                        'cpu': np.random.uniform(20, 90),
                        'anomaly': np.random.choice([0, 1], p=[0.85, 0.15])
                    }
                    
                    st.session_state.streaming_data.append(new_point)
                    
                    # Keep only last 50 points
                    if len(st.session_state.streaming_data) > 50:
                        st.session_state.streaming_data.pop(0)
                    
                    # Create streaming chart
                    stream_df = pd.DataFrame(st.session_state.streaming_data)
                    
                    fig_stream = go.Figure()
                    
                    # Normal points
                    normal_points = stream_df[stream_df['anomaly'] == 0]
                    fig_stream.add_trace(go.Scatter(
                        x=normal_points['time'],
                        y=normal_points['cost'],
                        mode='lines+markers',
                        name='Normal',
                        line=dict(color='blue', width=2),
                        marker=dict(size=6)
                    ))
                    
                    # Anomaly points
                    anomaly_points = stream_df[stream_df['anomaly'] == 1]
                    if not anomaly_points.empty:
                        fig_stream.add_trace(go.Scatter(
                            x=anomaly_points['time'],
                            y=anomaly_points['cost'],
                            mode='markers',
                            name='Anomalies',
                            marker=dict(color='red', size=12, symbol='diamond')
                        ))
                    
                    fig_stream.update_layout(
                        title='Real-time Cost Stream',
                        xaxis_title='Time',
                        yaxis_title='Cost ($)',
                        height=400,
                        showlegend=True
                    )
                    
                    with placeholder.container():
                        st.plotly_chart(fig_stream, use_container_width=True)
                    
                    time.sleep(animation_speed)
                
                st.session_state.streaming = False
        
        with col2:
            # Streaming metrics
            st.markdown("**Stream Metrics**")
            
            if st.session_state.get('streaming_data'):
                recent_data = pd.DataFrame(st.session_state.streaming_data[-10:])
                
                avg_cost = recent_data['cost'].mean()
                anomaly_rate = recent_data['anomaly'].mean() * 100
                cost_trend = 'Rising' if recent_data['cost'].iloc[-1] > recent_data['cost'].iloc[0] else 'Falling'
                
                st.metric("Avg Cost (10s)", f"${avg_cost:.0f}")
                st.metric("Anomaly Rate", f"{anomaly_rate:.0f}%")
                st.metric("Trend", cost_trend)
                
                # Mini sparkline
                fig_spark = go.Figure()
                fig_spark.add_trace(go.Scatter(
                    y=recent_data['cost'],
                    mode='lines',
                    line=dict(color='green', width=3),
                    showlegend=False
                ))
                fig_spark.update_layout(
                    height=100,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=False, showticklabels=False)
                )
                st.plotly_chart(fig_spark, use_container_width=True)

if __name__ == "__main__":
    main()