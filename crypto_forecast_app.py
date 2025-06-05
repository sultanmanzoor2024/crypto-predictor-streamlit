import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
from datetime import datetime, timedelta
from binance.client import Client
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, Conv1D, MaxPooling1D, Flatten, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go

client = Client()

# Constants
COINS = {
    "Bitcoin (BTC)": "BTCUSDT",
    "Ethereum (ETH)": "ETHUSDT",
    "Binance Coin (BNB)": "BNBUSDT",
    "Cardano (ADA)": "ADAUSDT",
    "Solana (SOL)": "SOLUSDT",
    "Ripple (XRP)": "XRPUSDT",
    "Dogecoin (DOGE)": "DOGEUSDT",
    "Polkadot (DOT)": "DOTUSDT",
    "Litecoin (LTC)": "LTCUSDT",
    "Avalanche (AVAX)": "AVAXUSDT"
}

MODEL_TYPES = ["LSTM", "GRU", "MLP", "CNN-LSTM"]

INTERVALS = {
    "Hourly": Client.KLINE_INTERVAL_1HOUR,
    "Daily": Client.KLINE_INTERVAL_1DAY,
    "Weekly": Client.KLINE_INTERVAL_1WEEK,
    "Monthly": Client.KLINE_INTERVAL_1MONTH,
}

MODEL_DIR = "./models"
SCALER_DIR = "./scalers"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

def fetch_data(symbol, interval):
    if interval == Client.KLINE_INTERVAL_1HOUR:
        lookback = "6 months ago UTC"
    else:
        lookback = "2 years ago UTC"
    klines = client.get_historical_klines(symbol, interval, lookback)
    df = pd.DataFrame(klines, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base", "Taker buy quote", "Ignore"
    ])
    df['Close'] = df['Close'].astype(float)
    df['Time'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('Time', inplace=True)
    return df[['Close']]

def prepare_data(data, window=60, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(data)
    else:
        scaled = scaler.transform(data)
    x, y = [], []
    for i in range(window, len(scaled)):
        x.append(scaled[i-window:i])
        y.append(scaled[i])
    x, y = np.array(x), np.array(y)
    return x, y, scaler

def build_model(model_type, input_shape, neurons=50, dropout=0.2, learning_rate=0.001):
    if model_type == "LSTM":
        model = Sequential([
            Input(shape=input_shape),
            LSTM(neurons, return_sequences=True),
            Dropout(dropout),
            LSTM(neurons),
            Dropout(dropout),
            Dense(1)
        ])
    elif model_type == "GRU":
        model = Sequential([
            Input(shape=input_shape),
            GRU(neurons, return_sequences=True),
            Dropout(dropout),
            GRU(neurons),
            Dropout(dropout),
            Dense(1)
        ])
    elif model_type == "MLP":
        model = Sequential([
            Input(shape=(input_shape[0]*input_shape[1],)),
            Dense(neurons*2, activation='relu'),
            Dropout(dropout),
            Dense(neurons, activation='relu'),
            Dense(1)
        ])
    elif model_type == "CNN-LSTM":
        model = Sequential([
            Input(shape=input_shape),
            TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')),
            TimeDistributed(MaxPooling1D(pool_size=2)),
            TimeDistributed(Flatten()),
            LSTM(neurons),
            Dropout(dropout),
            Dense(1)
        ])
    else:
        raise ValueError("Unknown model type")
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

def train_model(coin, model_type, interval, epochs=8, batch_size=32, neurons=50, dropout=0.2, lr=0.001):
    df = fetch_data(COINS[coin], INTERVALS[interval])
    scaler_path = os.path.join(SCALER_DIR, f"{coin}_{interval}_scaler.save")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = None
    x, y, scaler = prepare_data(df.values, scaler=scaler)
    if model_type == "MLP":
        x = x.reshape(x.shape[0], -1)
    model = build_model(model_type, x.shape[1:], neurons, dropout, lr)
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)
    model_path = os.path.join(MODEL_DIR, f"{coin}_{model_type}_{interval}.h5")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    return model, df, x, y, model_path, scaler_path

def evaluate_model(model, x, y, model_type):
    if model_type == "MLP":
        x = x.reshape(x.shape[0], -1)
    preds = model.predict(x)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    return mse, rmse

def full_future_prediction(model, last_seq, n_steps, model_type, scaler):
    preds = []
    current_seq = last_seq.copy()
    for _ in range(n_steps):
        input_seq = current_seq.reshape(1, *current_seq.shape)
        if model_type == "MLP":
            input_seq = input_seq.reshape(1, -1)
        pred_scaled = model.predict(input_seq)[0][0]
        preds.append(pred_scaled)
        current_seq = np.roll(current_seq, -1, axis=0)
        if model_type == "MLP":
            current_seq[-1] = pred_scaled
        else:
            current_seq[-1] = [pred_scaled]
    preds = np.array(preds).reshape(-1, 1)
    preds_inversed = scaler.inverse_transform(preds).flatten()
    return preds_inversed

st.title("🪙 Crypto Price Prediction: Multi-Coin, Models & Hyperparameters")

tab_train, tab_predict = st.tabs(["Train Multiple Models", "Predict Future Prices"])

with tab_train:
    st.header("Batch Train Models")
    coins_selected = st.multiselect("Select Coins to Train", list(COINS.keys()), default=["Bitcoin (BTC)", "Ethereum (ETH)"])
    model_type = st.selectbox("Model Type", MODEL_TYPES, index=0)
    interval = st.selectbox("Interval", list(INTERVALS.keys()), index=1)
    epochs = st.slider("Epochs", 1, 50, 8)
    batch_size = st.slider("Batch Size", 8, 128, 32)
    neurons = st.slider("Neurons per Layer", 10, 200, 50)
    dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
    learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")

    if st.button("Train Selected Models"):
        for coin in coins_selected:
            with st.spinner(f"Training {model_type} for {coin} ({interval}) ..."):
                model, df, x, y, model_path, scaler_path = train_model(
                    coin, model_type, interval, epochs, batch_size, neurons, dropout, learning_rate
                )
                mse, rmse = evaluate_model(model, x, y, model_type)
                st.success(f"Trained {coin} saved at {model_path}")
                st.write(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}")
                st.line_chart(df['Close'])

with tab_predict:
    st.header("Predict Future Prices")

    coin_pred = st.selectbox("Select Coin", list(COINS.keys()))
    model_pred = st.selectbox("Model Type", MODEL_TYPES)
    interval_pred = st.selectbox("Interval", list(INTERVALS.keys()))
    steps_pred = st.number_input("Number of Future Steps to Predict", min_value=1, max_value=365, value=30)

    model_path = os.path.join(MODEL_DIR, f"{coin_pred}_{model_pred}_{interval_pred}.h5")
    scaler_path = os.path.join(SCALER_DIR, f"{coin_pred}_{interval_pred}_scaler.save")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.warning("Model or scaler file not found! Please train the model first.")
    else:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        df = fetch_data(COINS[coin_pred], INTERVALS[interval_pred])
        x, y, _ = prepare_data(df.values, scaler=scaler)
        last_seq = x[-1]

        predicted_prices = full_future_prediction(model, last_seq, steps_pred, model_pred, scaler)

        last_date = df.index[-1]
        delta_map = {
            "Hourly": timedelta(hours=1),
            "Daily": timedelta(days=1),
            "Weekly": timedelta(weeks=1),
            "Monthly": timedelta(days=30)
        }
        delta = delta_map[interval_pred]
        future_dates = [last_date + delta * (i + 1) for i in range(steps_pred)]

        df_future = pd.DataFrame({"Date": future_dates, "Predicted Price (USD)": predicted_prices})

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical'))
        fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices, mode='lines+markers', name='Predicted'))
        fig.update_layout(title=f"Price Prediction for {coin_pred} ({model_pred})",
                          xaxis_title="Date", yaxis_title="Price (USD)", height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df_future)

st.markdown("---")
st.markdown("""
### Notes on Next Steps: Web Backend + React Frontend

- **Backend**: This entire training/prediction logic can be moved to a Flask/FastAPI backend exposing REST endpoints:
  - `/train` (POST) to trigger training with hyperparams
  - `/predict` (GET/POST) to get future price predictions
  - Save models and scalers on the server filesystem or cloud storage

- **Frontend**: React (or Next.js) SPA can:
  - Allow users to select coins, models, hyperparams for training
  - Display live training progress and results
  - Show prediction charts with interactive UI
  - Use WebSocket or polling to fetch updates

- **Advantages**:
  - Scalability and separation of concerns
  - Ability to deploy models on powerful GPU servers
  - Cleaner, more responsive UI with React ecosystem

If you want, I can help you scaffold the backend API with FastAPI + model training endpoints next!
""")
