import streamlit as st
import pandas as pd
import json

st.set_page_config(
    page_title="Brent Crude Oil Dashboard",
    layout="wide",
    page_icon="🛢️"
)

st.title("🛢️ Brent Crude Oil Price Forecasting Dashboard")

# Test data loading
try:
    df = pd.read_csv(
        "results/arima/arima_test_predictions.csv",
        parse_dates=["Date"],
        index_col="Date"
    )
    st.success(f"Data loaded: {len(df)} rows")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Error loading data: {e}")

tab1, tab2, tab3 = st.tabs([
    "📈 Model Performance",
    "🔍 All Models Comparison",
    "🔮 Future Forecast"
])

with tab1:
    st.write("Tab 1 is working")

with tab2:
    st.write("Tab 2 is working")

with tab3:
    st.write("Tab 3 is working")