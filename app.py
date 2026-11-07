import streamlit as st
import pandas as pd
import yfinance as yf
import subprocess
import sys
import re
from pathlib import Path

st.set_page_config(
    page_title="QuantFormer Stock Prediction",
    page_icon="📈",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

st.title("📈 QuantFormer - Stock Price Prediction")
st.write("Transformer-based stock market prediction dashboard")

st.sidebar.header("Stock Settings")

ticker = st.sidebar.selectbox(
    "Select Stock",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
)

period = st.sidebar.selectbox(
    "Historical Data Period",
    ["1y", "2y", "5y"]
)

if st.sidebar.button("Generate Prediction"):
    with st.spinner("Generating stock prediction..."):
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.predict",
                    "--ticker",
                    ticker,
                    "--period",
                    period
                ],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                match = re.search(
                    r"Estimated next-session close:\s*([\d.]+)",
                    result.stdout
                )

                if match:
                    st.session_state["prediction"] = float(match.group(1))
                    st.session_state["prediction_ticker"] = ticker
                    st.success("Prediction generated successfully!")
                else:
                    st.error("Prediction value not found.")
                    st.code(result.stdout)
            else:
                st.error("Prediction failed.")
                st.code(result.stderr)

        except Exception as e:
            st.error(str(e))

try:
    stock = yf.download(
        ticker,
        period=period,
        progress=False,
        auto_adjust=True
    )

    if stock.empty:
        st.error("No stock data available.")
        st.stop()

    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.get_level_values(0)

    latest_price = float(stock["Close"].iloc[-1])
    previous_price = float(stock["Close"].iloc[-2])
    price_change = latest_price - previous_price

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Latest Closing Price",
        f"₹{latest_price:.2f}",
        f"{price_change:.2f}"
    )

    col2.metric(
        "Previous Closing Price",
        f"₹{previous_price:.2f}"
    )

    col3.metric(
        "Available Records",
        len(stock)
    )

    st.subheader("Historical Stock Price")

    st.line_chart(stock["Close"])

    prediction = st.session_state.get("prediction")
    prediction_ticker = st.session_state.get("prediction_ticker")

    if prediction is not None and prediction_ticker == ticker:
        st.subheader("Next Trading Session Prediction")
        st.metric(
            "Estimated Closing Price",
            f"₹{prediction:.2f}"
        )
        st.info(
            "This is a model estimate, not a guaranteed market price."
        )
    else:
        st.info("Click 'Generate Prediction' in the sidebar.")

    prediction_file = OUTPUT_DIR / f"{ticker}_test_predictions.csv"

    if prediction_file.exists():
        predictions = pd.read_csv(prediction_file)

        st.subheader("Actual vs Predicted Prices")

        st.dataframe(predictions.tail(10), use_container_width=True)

        numeric_columns = predictions.select_dtypes(
            include="number"
        ).columns.tolist()

        if len(numeric_columns) >= 2:
            st.line_chart(
                predictions[numeric_columns[:2]]
            )

        if len(numeric_columns) >= 2:
            actual = predictions[numeric_columns[0]]
            predicted = predictions[numeric_columns[1]]

            mae = (actual - predicted).abs().mean()
            rmse = (((actual - predicted) ** 2).mean()) ** 0.5

            valid = actual != 0

            if valid.any():
                mape = (
                    (actual[valid] - predicted[valid]).abs()
                    / actual[valid].abs()
                ).mean() * 100

                st.subheader("Model Performance")

                m1, m2, m3 = st.columns(3)

                m1.metric("MAE", f"{mae:.2f}")
                m2.metric("RMSE", f"{rmse:.2f}")
                m3.metric("MAPE", f"{mape:.2f}%")

    chart_file = OUTPUT_DIR / f"{ticker}_predictions.png"

    if chart_file.exists():
        st.subheader("Prediction Performance Chart")
        st.image(str(chart_file), use_container_width=True)

except Exception as e:
    st.error(f"Unable to load stock information: {e}")