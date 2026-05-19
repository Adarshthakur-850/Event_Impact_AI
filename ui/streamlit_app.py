import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from io import BytesIO

# Add parent directory to path to allow importing analysis module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import preprocess, ttest, intervention, regression_model

st.set_page_config(page_title="Event Impact Model", layout="wide")

st.title("Event Impact Model")
st.markdown("Quantify the impact of an event on a time-series metric.")

uploaded_file = st.file_uploader("Upload CSV (columns: date, metric_value)", type=['csv'])

if uploaded_file is not None:
    try:
        # Load and Preview
        df = preprocess.load_data(uploaded_file)
        st.subheader("Data Preview")
        st.line_chart(df.set_index('date')['metric_value'])
        
        # Event Selector
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            event_date = st.date_input("Select Event Date", min_value=min_date, max_value=max_date, value=min_date + (max_date - min_date)//2)
        
        if st.button("Analyze Impact"):
            st.markdown("---")
            event_date_str = str(event_date)
            pre_event, post_event = preprocess.split_data(df, event_date_str)
            
            if pre_event.empty or post_event.empty:
                st.error("Selected event date must have data both before and after.")
            else:
                # 1. T-Test
                ttest_res = ttest.calculate_ttest(pre_event['metric_value'], post_event['metric_value'])
                
                # 2. ITS
                its_res = intervention.interrupted_time_series(df, event_date_str)
                df['counterfactual'] = its_res['predictions']
                
                # 3. Regression
                reg_res = regression_model.fit_regression(df, event_date_str)
                
                # Display Results
                st.header("Results Summary")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Pre-Event Mean", f"{ttest_res['pre_mean']:.2f}")
                c2.metric("Post-Event Mean", f"{ttest_res['post_mean']:.2f}")
                c3.metric("P-Value (T-Test)", f"{ttest_res['p_value']:.4f}")
                
                c4, c5, c6 = st.columns(3)
                c4.metric("Regression Event Coef", f"{reg_res['event_coef']:.2f}")
                c5.metric("ITS Impact Estimate", f"{its_res['impact']:.2f}")
                c6.metric("ITS Impact %", f"{its_res['impact_pct']:.2f}%")
                
                # Visualizations
                st.header("Visualizations")
                
                tab1, tab2 = st.tabs(["Time Series & Counterfactual", "Regression Summary"])
                
                with tab1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df['date'], y=df['metric_value'], name='Actual', mode='lines'))
                    fig.add_trace(go.Scatter(x=df['date'], y=df['counterfactual'], name='Counterfactual (Expected)', line=dict(dash='dash')))
                    fig.add_vline(x=event_date_str, line_width=2, line_dash="dash", line_color="red", annotation_text="Event")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab2:
                    st.text(reg_res['summary'])
    
    except Exception as e:
        st.error(f"Error: {e}")
