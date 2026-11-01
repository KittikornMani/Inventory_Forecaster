"""Inventory Demand Forecaster Streamlit app."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting import forecast, load_artifacts, validate_upload
from inventory import recommend

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="Inventory Demand Forecaster", page_icon="📦", layout="wide")
st.title("Inventory Demand Forecaster")
st.caption("Single product-store M5 experiment · LSTM and GRU · 7–28 day demand")

with st.sidebar:
    st.header("Forecast settings")
    model_name = st.selectbox("Model", ["LSTM", "GRU"])
    horizon = st.selectbox("Days to forecast", [7, 14, 28], index=2)
    current_stock = st.number_input("Current stock (units)", min_value=0, value=50, step=1)
    lead_days = st.number_input("Supplier lead time (days)", min_value=1, max_value=28, value=7)
    safety_pct = st.number_input("Safety buffer (%)", min_value=0, max_value=100, value=20)
    st.divider()
    uploaded = st.file_uploader("Daily sales CSV", type="csv")
    future_file = st.file_uploader("Future calendar CSV (optional)", type="csv")
    run = st.button("Generate forecast", type="primary", use_container_width=True)

with st.expander("CSV format and model scope"):
    st.write(
        "Upload consecutive daily rows with: `series_id,date,sales,sell_price,is_event,snap`. "
        "Use the product-store series the models were trained on. "
        "`is_event` and `snap` are 0 or 1. The optional future calendar has "
        "`date,is_event,snap`. Without it, future event and SNAP flags are assumed 0. "
        "The last known selling price is carried forward."
    )
    st.write("At least 84 days are required. See `sample_data/upload_template.csv` for the column layout.")

info_path = MODEL_DIR / "model_info.json"
if not info_path.is_file():
    st.error("Setup needed: add models/model_info.json with the exact trained series_id. See README.md.")
    st.stop()
try:
    model_info = json.loads(info_path.read_text(encoding="utf-8"))
    trained_id = str(model_info["series_id"])
    if not trained_id:
        raise ValueError("series_id is empty")
except (ValueError, KeyError) as error:
    st.error(f"Invalid model_info.json: {error}")
    st.stop()

st.info(f"Trained series: `{trained_id}`. Forecasts for other products require separate training.")
if not run:
    st.write("Upload a matching daily sales CSV, then select **Generate forecast**.")
    st.stop()
if uploaded is None:
    st.warning("Upload a daily sales CSV first.")
    st.stop()

try:
    raw = validate_upload(pd.read_csv(uploaded), trained_id)
    future_calendar = None
    if future_file is not None:
        future_calendar = pd.read_csv(future_file)
        needed = {"date", "is_event", "snap"}
        missing = needed - set(future_calendar.columns)
        if missing:
            raise ValueError("Future calendar missing: " + ", ".join(sorted(missing)))
        future_calendar["date"] = pd.to_datetime(future_calendar["date"], errors="coerce")
        if future_calendar["date"].isna().any() or future_calendar["date"].duplicated().any():
            raise ValueError("Future calendar dates must be valid and unique.")
        for column in ("is_event", "snap"):
            future_calendar[column] = pd.to_numeric(future_calendar[column], errors="coerce")
            if not future_calendar[column].isin([0, 1]).all():
                raise ValueError(f"Future {column} must be 0 or 1 on every row.")
        expected = pd.date_range(raw["date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon)
        if not expected.isin(future_calendar["date"]).all():
            raise ValueError("Future calendar must cover every forecast date.")

    models, feature_scaler, target_scaler, window_size, feature_columns, log_target = load_artifacts(MODEL_DIR)
    prediction = forecast(
        raw, models[model_name], feature_scaler, target_scaler, horizon,
        future_calendar, window_size, feature_columns, log_target,
    )
    inventory = recommend(prediction, current_stock, lead_days, safety_pct)
except (ValueError, FileNotFoundError, KeyError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(f"Could not generate a forecast: {error}")
    st.stop()

st.subheader("Forecast overview")
if not log_target:
    st.caption("Loaded the supplied 28-day, 7-feature model format. Its saved test WAPE was about 119%, so treat recommendations as exploratory.")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Predicted demand", f"{inventory['forecasted_demand']:.1f} units")
k2.metric("Daily average", f"{inventory['average_daily_demand']:.2f} units")
k3.metric("Reorder point", f"{inventory['reorder_point']:.1f} units")
k4.metric("Suggested order", f"{inventory['suggested_order_quantity']} units")

if inventory["risk"] == "Reorder now":
    st.warning("Current stock is at or below the calculated reorder point.")
else:
    st.success("Current stock is above the calculated reorder point.")
if inventory["first_reorder_date"] is not None:
    st.write("Projected reorder threshold date:", inventory["first_reorder_date"].date())

tab_forecast, tab_stock, tab_details = st.tabs(["Demand forecast", "Inventory", "Details & downloads"])
with tab_forecast:
    figure = go.Figure()
    recent = raw.tail(56)
    figure.add_scatter(x=recent["date"], y=recent["sales"], name="Historical sales", mode="lines+markers")
    figure.add_scatter(x=prediction["date"], y=prediction["predicted_sales"], name=f"{model_name} forecast", mode="lines+markers")
    figure.update_layout(xaxis_title="Date", yaxis_title="Units sold per day", hovermode="x unified")
    st.plotly_chart(figure, use_container_width=True)
    st.caption("Fractional units are expected demand estimates. Actual sales occur in whole units.")

with tab_stock:
    stock = pd.DataFrame({
        "date": pd.concat([pd.Series([raw["date"].iloc[-1]]), prediction["date"]], ignore_index=True),
        "remaining": [float(current_stock)] + inventory["remaining_stock"].tolist(),
    })
    figure = go.Figure()
    figure.add_scatter(x=stock["date"], y=stock["remaining"], name="Projected stock", mode="lines+markers")
    figure.add_hline(y=inventory["reorder_point"], line_dash="dash", line_color="red", annotation_text="Reorder point")
    figure.add_hline(y=0, line_dash="dot", line_color="gray")
    figure.update_layout(xaxis_title="Date", yaxis_title="Units in stock")
    st.plotly_chart(figure, use_container_width=True)
    st.caption("Projection assumes no replenishment during the forecast. Lead time and safety buffer are user inputs.")

with tab_details:
    metrics = model_info.get("test_metrics", {})
    if metrics:
        st.write("Held-out test performance (lower is better)")
        scores = pd.DataFrame(metrics).T
        scores["WAPE (%)"] = scores.pop("WAPE") * 100
        st.dataframe(scores.round(3), use_container_width=True)
    else:
        st.caption("Test metrics will appear after adding a model_info.json from the updated Kaggle run.")
    st.dataframe(prediction, use_container_width=True, hide_index=True)
    st.download_button("Download forecast CSV", prediction.to_csv(index=False), "forecast_results.csv", "text/csv")
    summary = pd.DataFrame([{
        "series_id": trained_id,
        "model": model_name,
        "current_inventory": current_stock,
        "average_daily_demand": inventory["average_daily_demand"],
        "forecasted_demand": inventory["forecasted_demand"],
        "safety_stock": inventory["safety_stock"],
        "reorder_point": inventory["reorder_point"],
        "suggested_order_quantity": inventory["suggested_order_quantity"],
        "risk": inventory["risk"],
    }])
    st.download_button("Download inventory recommendation", summary.to_csv(index=False), "inventory_recommendation.csv", "text/csv")
    st.caption("The current model was evaluated on one series; earlier test WAPE was above 100%. Use recommendations as a research demonstration.")
