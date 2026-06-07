import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(
    page_title="Brent Crude Oil Price Forecasting Dashboard",
    layout="wide",
    page_icon="🛢️"
)

FILE_MAP = {
    "ARIMA":         ("results/arima/arima_test_predictions.csv",
                      "results/arima/arima_metrics.json"),
    "Random Forest": ("results/random_forest/rf_test_predictions.csv",
                      "results/random_forest/rf_metrics.json"),
    "XGBoost":       ("results/xgboost/xgb_test_predictions.csv",
                      "results/xgboost/xgb_metrics.json"),
    "LSTM":          ("results/lstm/lstm_test_predictions.csv",
                      "results/lstm/lstm_metrics.json"),
    "GRU":           ("results/gru/gru_test_predictions.csv",
                      "results/gru/gru_metrics.json"),
}

COLOURS = {
    "ARIMA":         "grey",
    "Random Forest": "steelblue",
    "XGBoost":       "darkorange",
    "LSTM":          "crimson",
    "GRU":           "seagreen",
}

@st.cache_data
def load_predictions(model_name):
    path, _ = FILE_MAP[model_name]
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df

@st.cache_data
def load_metrics(model_name):
    _, path = FILE_MAP[model_name]
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_all():
    preds, metrics = {}, {}
    for name in FILE_MAP:
        preds[name]   = load_predictions(name)
        metrics[name] = load_metrics(name)
    return preds, metrics

st.sidebar.title("🛢️ Brent Crude Oil")
st.sidebar.markdown("**MSc Data Science — UWE Bristol**")
st.sidebar.markdown(
    "Comparative evaluation of ARIMA, Random Forest, "
    "XGBoost, LSTM and GRU on 25 years of Brent crude "
    "oil futures data.")
st.sidebar.markdown("---")

selected_model = st.sidebar.selectbox(
    "Select model", list(FILE_MAP.keys()))

st.sidebar.markdown("**Date range (test period)**")
start_date = st.sidebar.date_input(
    "From", value=pd.Timestamp("2023-01-03"))
end_date = st.sidebar.date_input(
    "To", value=pd.Timestamp("2024-12-31"))

horizon = st.sidebar.radio(
    "Future forecast horizon",
    ["7 days", "30 days", "90 days"])
horizon_days = int(horizon.split()[0])

preds, metrics = load_all()

df_sel = load_predictions(selected_model)
mask = ((df_sel.index >= pd.Timestamp(start_date)) &
        (df_sel.index <= pd.Timestamp(end_date)))
df_fil = df_sel[mask]

st.title("🛢️ Brent Crude Oil Price Forecasting Dashboard")
st.markdown("Test period: **3 January 2023 – 31 December 2024** ")

try:
    tab1, tab2, tab3 = st.tabs(["Model Performance", "All Models Comparison", "Future Forecast"])
    
    with tab1:
        st.header(f"Model Performance – {selected_model}")
        
        if selected_model in metrics and metrics[selected_model]:
            m = metrics[selected_model]
            rmse = m.get("RMSE", np.nan)
            mae = m.get("MAE", np.nan)
            mape = m.get("MAPE", np.nan)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE ($/barrel)", f"{rmse:.3f}" if not np.isnan(rmse) else "N/A")
            c2.metric("MAE ($/barrel)", f"{mae:.3f}" if not np.isnan(mae) else "N/A")
            c3.metric("MAPE (%)", f"{mape:.2f}%" if not np.isnan(mape) else "N/A")
        
        if not df_fil.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_fil.index, y=df_fil["Actual"],
                mode="lines", name="Actual",
                line=dict(color="darkblue", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=df_fil.index, y=df_fil["Predicted"],
                mode="lines", name="Predicted",
                line=dict(color=COLOURS.get(selected_model, "darkorange"), dash="dash", width=2)
            ))
            fig.update_layout(
                title=f"{selected_model} — Actual vs Predicted",
                xaxis_title="Date",
                yaxis_title="Price (USD/barrel)",
                height=450,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            residuals = df_fil["Actual"] - df_fil["Predicted"]
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=df_fil.index, y=residuals,
                mode="lines", name="Residuals",
                line=dict(color="firebrick")
            ))
            fig_res.update_layout(
                title="Residuals (Actual - Predicted)",
                xaxis_title="Date",
                yaxis_title="Residual",
                height=350,
                hovermode="x unified"
            )
            st.plotly_chart(fig_res, use_container_width=True)
        else:
            st.warning("No data available for selected date range.")
    
    with tab2:
        st.header("All Models Comparison")
        
        metrics_rows = []
        for name in FILE_MAP.keys():
            if name in metrics and metrics[name]:
                m = metrics[name]
                metrics_rows.append({
                    "Model": name,
                    "RMSE": m.get("RMSE", np.nan),
                    "MAE": m.get("MAE", np.nan),
                    "MAPE": m.get("MAPE", np.nan)
                })
        
        if metrics_rows:
            metrics_df = pd.DataFrame(metrics_rows)
            metrics_df = metrics_df.sort_values("RMSE", ascending=True).reset_index(drop=True)
            metrics_df["Rank"] = range(1, len(metrics_df) + 1)
            st.dataframe(metrics_df.style.format({"RMSE": "{:.3f}", "MAE": "{:.3f}", "MAPE": "{:.2f}"}), use_container_width=True)
        
        fig_all = go.Figure()
        actual_added = False
        for name in FILE_MAP.keys():
            if name in preds:
                df_pred = preds[name]
                if not actual_added:
                    fig_all.add_trace(go.Scatter(
                        x=df_pred.index, y=df_pred["Actual"],
                        mode="lines", name="Actual",
                        line=dict(color="black", width=3)
                    ))
                    actual_added = True
                fig_all.add_trace(go.Scatter(
                    x=df_pred.index, y=df_pred["Predicted"],
                    mode="lines", name=name,
                    line=dict(color=COLOURS.get(name, "gray"))
                ))
        
        fig_all.update_layout(
            title="All Models — Predictions vs Actual",
            xaxis_title="Date",
            yaxis_title="Price (USD/barrel)",
            height=500,
            hovermode="x unified"
        )
        st.plotly_chart(fig_all, use_container_width=True)
    
    with tab3:
        st.header("Future Forecast")
        st.write(f"Forecast horizon: **{horizon_days} days**")
        
        if selected_model in preds and not preds[selected_model].empty:
            last_idx = preds[selected_model].index[-1]
            last_pred = float(preds[selected_model]["Predicted"].iloc[-1])
            
            future_dates = pd.date_range(start=last_idx + timedelta(days=1), periods=horizon_days, freq="D")
            future_vals = [last_pred] * horizon_days
            
            fig_fut = go.Figure()
            fig_fut.add_trace(go.Scatter(
                x=preds[selected_model].index, y=preds[selected_model]["Actual"],
                mode="lines", name="Actual (Test)",
                line=dict(color="darkblue", width=2)
            ))
            fig_fut.add_trace(go.Scatter(
                x=preds[selected_model].index, y=preds[selected_model]["Predicted"],
                mode="lines", name="Predicted (Test)",
                line=dict(color=COLOURS.get(selected_model, "darkorange"), width=2)
            ))
            fig_fut.add_trace(go.Scatter(
                x=future_dates, y=future_vals,
                mode="lines+markers", name="Future Forecast",
                line=dict(color=COLOURS.get(selected_model, "darkorange"), dash="dot", width=2)
            ))
            fig_fut.update_layout(
                title=f"{selected_model} — Future Forecast ({horizon_days} days)",
                xaxis_title="Date",
                yaxis_title="Price (USD/barrel)",
                height=450,
                hovermode="x unified"
            )
            st.plotly_chart(fig_fut, use_container_width=True)
            
            forecast_table = pd.DataFrame({
                "Date": future_dates,
                "Forecast (USD/barrel)": future_vals
            })
            st.dataframe(forecast_table, use_container_width=True)
        else:
            st.warning(f"No prediction data available for {selected_model}.")

except Exception as e:
    st.error(f"Error loading dashboard: {str(e)}")
    st.write("Debug info:")
    st.write(f"Selected model: {selected_model}")
    st.write(f"Available models in FILE_MAP: {list(FILE_MAP.keys())}")
