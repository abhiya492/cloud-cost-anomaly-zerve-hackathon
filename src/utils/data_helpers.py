import streamlit as st
import pandas as pd
import plotly.express as px

def show_data_summary(df: pd.DataFrame):
    """Display comprehensive data summary for user understanding"""
    
    if df.empty:
        st.warning("No data available to analyze")
        return
    
    # Data Quality Indicators
    st.markdown("### 📊 **Data Quality & Coverage**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%", 
                 help="Percentage of non-missing values in dataset")
    
    with col2:
        cost_range = df['cost'].max() - df['cost'].min()
        st.metric("Cost Range", f"${cost_range:.0f}", 
                 help="Difference between highest and lowest costs")
    
    with col3:
        avg_utilization = df.get('cpu_utilization', pd.Series([0])).mean()
        st.metric("Avg CPU Usage", f"{avg_utilization:.1f}%",
                 help="Average CPU utilization across all resources")
    
    with col4:
        efficiency_score = (avg_utilization / 100) * (completeness / 100) * 100
        st.metric("Efficiency Score", f"{efficiency_score:.0f}%",
                 help="Overall resource efficiency rating")

def show_anomaly_explanation():
    """Explain how anomaly detection works"""
    
    st.markdown("### 🤖 **How Anomaly Detection Works**")
    
    with st.expander("Click to understand the AI model"):
        st.markdown("""
        **Our ML Model Uses:**
        
        1. **🔍 Isolation Forest Algorithm**
           - Identifies unusual patterns in cost data
           - Looks for outliers that don't fit normal behavior
        
        2. **📊 Key Features Analyzed:**
           - Cost amount and trends
           - CPU utilization patterns  
           - Usage hours and efficiency
           - Time-based patterns (weekday/weekend)
           - Service and environment context
        
        3. **🎯 Confidence Scoring:**
           - **90-100%**: Very confident anomaly
           - **70-89%**: Likely anomaly  
           - **50-69%**: Possible anomaly
           - **<50%**: Low confidence
        """)