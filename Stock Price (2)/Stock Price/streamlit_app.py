import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load model
model = load_model('stock.h5')

# Page configuration
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor")

# Add header image
st.image("https://tse1.mm.bing.net/th?id=OIP.Xy8NWJ2DaTHHoEWQW26QiAHaFx&pid=Api", 
         caption="Stock Market Analysis", use_column_width=True)

# Input
ticker = st.text_input("Enter stock symbol (e.g., AAPL):")

if ticker:
    # Get today's date
    today = datetime.today().strftime('%Y-%m-%d')

    # Download up-to-date stock data
    df = yf.download(ticker, start='2010-01-01', end=today)

    if len(df) < 60:
        st.warning("Not enough data available for prediction.")
    else:
        data = df[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Last 60 days for prediction
        last_seq = scaled_data[-60:]
        input_seq = last_seq.reshape(1, 60, 1)

        # Predict next 7 days
        predictions = []
        for _ in range(7):
            pred = model.predict(input_seq)[0][0]
            predictions.append(pred)
            input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

        # Convert predictions back to original scale
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        # Generate future dates starting from today
        today_date = datetime.today().date()
        future_dates = [today_date + timedelta(days=i+1) for i in range(7)]

        # Plot results
        fig, ax = plt.subplots()
        ax.plot(future_dates, predictions, marker='o', linestyle='--', color='green')
        ax.set_title(f"Next 7-Day Prediction for {ticker}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Predicted Price ($)")
        ax.grid(True)
        st.pyplot(fig)

        # Show predicted values
        st.subheader("📊 Forecast Summary:")
        for date, price in zip(future_dates, predictions.flatten()):
            st.write(f"📅 {date.strftime('%Y-%m-%d')} → 💲 {price:.2f}")

        # Add a second stock-themed image
        st.image("https://tse1.mm.bing.net/th?id=OIP.9CdnNc5s0bBGCRHAICZsgwHaE7&pid=Api", 
                 caption="Forecasting Stock Price Trends", use_column_width=True)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and TensorFlow")
