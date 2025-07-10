import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

# --- Configuration ---
ticker = 'AAPL'
start_date = '2015-01-01'
end_date = '2023-12-31'
model_path = 'lstm_stock_model.h5'
sequence_length = 60

# --- Step 1: Load Data ---
df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

if df.empty:
    raise ValueError(f"❌ No data fetched for ticker '{ticker}'.")

print("Available columns:", df.columns)
if 'Close' not in df.columns:
    raise ValueError(f"❌ 'Close' column missing in returned data: {df.columns}")

# Remove NaN values
data = df[['Close']].dropna()
if data.empty:
    raise ValueError("❌ 'Close' column has only NaN values.")

# --- Step 2: Preprocess Data ---
dataset = data.values  # shape (n, 1)
print("✅ Dataset shape before scaling:", dataset.shape)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_len = int(len(dataset) * 0.8)
train_data = scaled_data[:train_len]
test_data = scaled_data[train_len - sequence_length:]

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# --- Step 3: Load or Train Model ---
if os.path.exists(model_path):
    print("📥 Found saved model. Loading it...")
    model = load_model(model_path)
else:
    print("🔧 No saved model found. Building and training a new one...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=10)
    model.save(model_path)
    print(f"✅ Model trained and saved as '{model_path}'")

# --- Step 4: Predict Prices ---
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

# --- Step 5: Plot Results ---
train = data[:train_len]
valid = data[train_len:].copy()
valid['Predictions'] = predictions

plt.figure(figsize=(14,6))
plt.title(f'{ticker} Stock Price Prediction (LSTM)')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.plot(train['Close'], label='Training Data')
plt.plot(valid['Close'], label='Actual Price')
plt.plot(valid['Predictions'], label='Predicted Price')
plt.legend()
plt.show()

def predict_stock_lstm(ticker='AAPL', start_date='2015-01-01', end_date='2023-12-31'):
    model_path = f"model_{ticker}.h5"
    sequence_length = 60

    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    if df.empty or 'Close' not in df.columns:
        return None, None, f"❌ No valid 'Close' price data for '{ticker}'."

    data = df[['Close']].dropna()
    if data.empty:
        return None, None, f"❌ 'Close' column only contains NaNs."

    dataset = data.values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(dataset)

    train_len = int(len(dataset) * 0.8)
    train_data = scaled_data[:train_len]
    test_data = scaled_data[train_len - sequence_length:]

    def create_sequences(data, seq_length):
        x, y = [], []
        for i in range(seq_length, len(data)):
            x.append(data[i - seq_length:i])
            y.append(data[i])
        return np.array(x), np.array(y)

    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(50),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)
        model.save(model_path)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    valid = data[train_len:].copy()
    valid['Predictions'] = predictions

    return valid, data, None