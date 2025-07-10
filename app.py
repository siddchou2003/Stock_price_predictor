import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from stock_predict_lstm import predict_stock_lstm

st.set_page_config(page_title="LSTM Stock Price Predictor", layout="wide")

st.title("📈 LSTM Stock Price Predictor")

# Inputs
ticker = st.text_input("Enter Stock Ticker", "AAPL")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
with col2:
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31"))

# Predict Button
if st.button("Predict"):
    with st.spinner("🔄 Fetching and predicting..."):
        valid, full_data, error = predict_stock_lstm(
            ticker=ticker.upper(),
            start_date=str(start_date),
            end_date=str(end_date)
        )

        if error:
            st.error(error)
        else:
            st.success("✅ Prediction complete.")
            st.subheader(f"{ticker.upper()} Price Forecast")

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(full_data['Close'], label="Training Data")
            ax.plot(valid['Close'], label="Actual Price", color='orange')
            ax.plot(valid['Predictions'], label="Predicted Price", color='green')
            ax.set_xlabel("Date")
            ax.set_ylabel("Close Price")
            ax.set_title(f"{ticker.upper()} Stock Price Prediction")
            ax.legend()
            st.pyplot(fig)