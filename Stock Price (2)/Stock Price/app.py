from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
model = load_model('stock.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker'].upper()
    df = yf.download(ticker, start='2010-01-01', end='2024-01-01')
    
    if df.empty or len(df) < 60:
        return render_template('index.html', error="Not enough data for prediction.")

    data = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create last 60-day sequence
    last_seq = scaled_data[-60:]
    input_seq = last_seq.reshape(1, 60, 1)

    # Predict next 7 days
    predictions = []
    for _ in range(7):
        pred = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    dates = [(df.index[-1] + datetime.timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(7)]

    # Plot prediction
    plt.figure(figsize=(8, 4))
    plt.plot(dates, predictions, marker='o', linestyle='--', color='green')
    plt.title(f'Next 7-Day Prediction for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot to static folder
    plot_path = f'static/pred_{ticker}.png'
    plt.savefig(plot_path)
    plt.close()

    return render_template('result.html', ticker=ticker, dates=dates, predictions=predictions.flatten(), plot_url=plot_path)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
