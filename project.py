from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Stock Price Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request Models ───────────────────────────────────────────────
class PredictRequest(BaseModel):
    ticker: str
    period: str = "2y"       # Data period: 1y, 2y, 5y
    future_days: int = 30    # Days to predict ahead

# ─── ANN Model Builder ────────────────────────────────────────────
def build_ann_model(input_dim: int) -> Sequential:
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# ─── Feature Engineering ─────────────────────────────────────────
def create_features(df: pd.DataFrame, lookback: int = 10) -> tuple:
    df = df.copy()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['EMA_10'] = df['Close'].ewm(span=10).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['Volume_MA'] = df['Volume'].rolling(10).mean()
    df['Price_Change'] = df['Close'].pct_change()
    df['Volatility'] = df['Close'].rolling(10).std()
    df['High_Low_Range'] = df['High'] - df['Low']

    for lag in range(1, lookback + 1):
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)

    df.dropna(inplace=True)

    feature_cols = [
        'SMA_10', 'SMA_20', 'EMA_10', 'RSI',
        'Volume_MA', 'Price_Change', 'Volatility', 'High_Low_Range'
    ] + [f'Close_Lag_{i}' for i in range(1, lookback + 1)]

    X = df[feature_cols].values
    y = df['Close'].values
    return X, y, df

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

# ─── Prediction Endpoint ─────────────────────────────────────────
@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        # 1. Fetch data
        stock = yf.Ticker(req.ticker.upper())
        df = stock.history(period=req.period)
        if df.empty or len(df) < 60:
            raise HTTPException(status_code=400, detail=f"Not enough data for ticker '{req.ticker}'")

        df.reset_index(inplace=True)
        dates = df['Date'].astype(str).tolist()

        # 2. Feature engineering
        X, y, df_feat = create_features(df)

        # 3. Scale
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # 4. Train/test split (80/20)
        split = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y_scaled[:split], y_scaled[split:]

        # 5. Build & train ANN
        model = build_ann_model(X_scaled.shape[1])
        es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=150,
            batch_size=32,
            callbacks=[es],
            verbose=0
        )

        # 6. Predictions on full dataset
        y_pred_scaled = model.predict(X_scaled, verbose=0).flatten()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_actual = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

        # 7. Metrics
        rmse = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
        mae  = float(mean_absolute_error(y_actual, y_pred))
        r2   = float(r2_score(y_actual, y_pred))
        accuracy = float(max(0, 100 - (rmse / np.mean(y_actual) * 100)))

        # 8. Future prediction (iterative)
        last_X = X_scaled[-1].copy()
        future_prices = []
        for _ in range(req.future_days):
            pred_s = model.predict(last_X.reshape(1, -1), verbose=0)[0][0]
            pred_price = float(scaler_y.inverse_transform([[pred_s]])[0][0])
            future_prices.append(round(pred_price, 2))
            # shift lags
            lag_start = X_scaled.shape[1] - 10
            last_X[lag_start+1:] = last_X[lag_start:-1]
            last_X[lag_start] = pred_s

        # 9. Training loss history
        train_loss = [round(float(v), 6) for v in history.history['loss']]
        val_loss   = [round(float(v), 6) for v in history.history['val_loss']]

        feat_dates = df_feat['Date'].astype(str).tolist()

        return {
            "ticker": req.ticker.upper(),
            "actual_prices": [round(float(p), 2) for p in y_actual],
            "predicted_prices": [round(float(p), 2) for p in y_pred],
            "dates": feat_dates,
            "future_prices": future_prices,
            "metrics": {
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "r2": round(r2, 4),
                "accuracy": round(accuracy, 2)
            },
            "train_loss": train_loss,
            "val_loss": val_loss,
            "current_price": round(float(y_actual[-1]), 2),
            "predicted_next": future_prices[0] if future_prices else None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Stock Price Prediction API is running!"}

@app.get("/health")
async def health():
    return {"status": "ok"}
