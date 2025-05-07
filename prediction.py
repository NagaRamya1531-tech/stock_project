import os
import time
import math
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf

import matplotlib
matplotlib.use('Agg')  # 👈 forces non-GUI backend
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sentiment import get_multi_social_sentiment, make_recommendation
try:
    from alpha_vantage.timeseries import TimeSeries
except ImportError:
    TimeSeries = None


def get_historical(quote: str, fallback_key: str = None, csv_dir: str = ".") -> pd.DataFrame:
    os.makedirs(csv_dir, exist_ok=True)
    cache_path = os.path.join(csv_dir, f"{quote}.csv")

    # 1. Try cache first
    if os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if 'Close' in cached.columns:
                print(f"[INFO] Loaded {quote} from cache.")
                return cached
        except Exception as e:
            print(f"[WARN] Failed to load cache: {e}")

    df = pd.DataFrame()
    start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")

    # 2. Try yfinance with retries
    for attempt in range(1, 4):
        try:
            tmp = yf.download(quote, start=start, end=end, auto_adjust=False, progress=False)
            if isinstance(tmp, pd.DataFrame) and not tmp.empty:
                df = tmp.copy()
                print(f"[INFO] Fetched {quote} from yfinance (attempt {attempt})")
                break
        except Exception as e:
            print(f"[ERROR] yfinance attempt {attempt} failed: {e}")
            time.sleep(5 * attempt)

    # 3. Fallback to Alpha Vantage (free tier using get_daily)
    if df.empty and fallback_key and TimeSeries:
        try:
            ts = TimeSeries(key=fallback_key, output_format='pandas')
            data, _ = ts.get_daily(symbol=quote, outputsize='full')  # ✅ FREE API
            data = (
                data.head(503).iloc[::-1].reset_index().rename(columns={
                    'date': 'Date',
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. volume': 'Volume'
                })
            )
            data['Adj Close'] = data['Close']  # Create dummy Adj Close
            df = data.set_index('Date')[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            print(f"[INFO] Fetched {quote} from Alpha Vantage")
        except Exception as e:
            print(f"[ERROR] Alpha Vantage fallback failed: {e}")

    # 4. Final fallback to cache if still empty
    if df.empty and os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if 'Close' in cached.columns:
                print(f"[INFO] Fallback to cached {quote}")
                return cached
        except Exception:
            pass

    # 5. Still empty — raise error
    if df.empty:
        raise RuntimeError(f"❌ Could not fetch ANY data for {quote}")

    # 6. Clean & cache
    df = df.dropna(subset=['Close'])
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        df[col] = pd.to_numeric(df.get(col, np.nan), errors='coerce')
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    df = df.dropna(subset=['Close'])
    df.to_csv(cache_path)
    print(f"[INFO] Saved cleaned {quote} data to cache.")

    return df


# ─── 1. Trend Figure ───────────────────────────────────────────────────────────
def create_trend_fig(df: pd.DataFrame) -> plt.Figure:
    """
    Returns a Matplotlib Figure showing the historical closing price trend.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df.index, df['Close'], label='Close Price')
    ax.set_title("Historical Closing Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    return fig


# ─── 2. ARIMA ──────────────────────────────────────────────────────────────────
def ARIMA_ALGO(df: pd.DataFrame) -> (float, float):
    """
    Returns (Day 1 Forecast, RMSE from past data)
    """
    series = df["Close"].values
    split = int(len(series) * 0.8)
    train, test = series[:split], series[split:]
    history = list(train)
    preds = []
    for obs in test:
        model = ARIMA(history, order=(6,1,0))
        fit = model.fit()
        yhat = fit.forecast()[0]
        preds.append(yhat)
        history.append(obs)
    rmse = math.sqrt(mean_squared_error(test, preds))

    # 🔁 REPLACE return with value from forecast_arima_7()
    arima7_list = forecast_arima_7(df)  # reuse existing full-model forecast
    tomorrow = arima7_list[0] if arima7_list else preds[-2]

    return tomorrow, rmse



def create_arima_fig(df: pd.DataFrame) -> plt.Figure:
    """
    Returns a Matplotlib Figure of actual vs predicted for the ARIMA model.
    """
    series = df["Close"].values
    split = int(len(series) * 0.8)
    train, test = series[:split], series[split:]
    history = list(train)
    preds = []
    for obs in test:
        model = ARIMA(history, order=(6,1,0))
        fit = model.fit()
        yhat = fit.forecast()[0]
        preds.append(yhat)
        history.append(obs)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(test, label="Actual Price")
    ax.plot(preds, label="Predicted Price")
    ax.set_title("ARIMA Model Fit")
    ax.legend()
    return fig


# ─── 3. LSTM ───────────────────────────────────────────────────────────────────
def LSTM_ALGO(df: pd.DataFrame) -> (float, float, list):
    """
    Returns (tomorrow forecast, RMSE, 7-day forecast list)
    """
    data = df["Close"].values.reshape(-1, 1)
    split = int(len(data) * 0.8)
    train = data[:split]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    # Create sequences
    X, y = [], []
    for i in range(100, len(scaled)):
        X.append(scaled[i-100:i])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)

    # Build the LSTM model
    from keras.models import Sequential
    from keras.layers import LSTM, Dropout, Dense, Input

    model = Sequential([
        Input(shape=(100, 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.1),
        LSTM(50),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=25, batch_size=32, verbose=0)

    # Predict for evaluation
    pred_scaled = model.predict(X, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)
    actual = scaler.inverse_transform(y.reshape(-1, 1))
    rmse = math.sqrt(mean_squared_error(actual, pred))

    # Predict 7 future days
    last_seq = scaled[-100:].reshape(1, 100, 1)
    forecast_scaled = []
    for _ in range(7):
        next_pred = model.predict(last_seq, verbose=0)[0, 0]
        forecast_scaled.append(next_pred)

        # update sequence with the predicted value
        last_seq = np.append(last_seq[:, 1:, :], [[[next_pred]]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

    return forecast[0], rmse, list(forecast)



def create_lstm_fig(df: pd.DataFrame) -> plt.Figure:
    """
    Returns a Matplotlib Figure of actual vs predicted for the LSTM model.
    """
    # reuse the same train/test split code
    data = df["Close"].values.reshape(-1, 1)
    split = int(len(data) * 0.8)
    train, test = data[:split], data[split-7:]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(np.vstack((train, test)))

    X, y = [], []
    for i in range(7, len(scaled)):
        X.append(scaled[i-7:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    from keras.models import Sequential
    from keras.layers import LSTM, Dropout, Dense

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(7,1)),
        Dropout(0.1),
        LSTM(50),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=25, batch_size=32, verbose=0)

    pred_scaled = model.predict(X, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)
    actual = scaler.inverse_transform(y.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(actual, label="Actual Price")
    ax.plot(pred, label="Predicted Price")
    ax.set_title("LSTM Model Fit")
    ax.legend()
    return fig


# ─── 4. Linear Regression ─────────────────────────────────────────────────────
def LIN_REG_ALGO(df: pd.DataFrame) -> (float, float):
    df2 = df[["Close"]].copy()
    df2["Future"] = df2["Close"].shift(-7)
    df2.dropna(inplace=True)

    X = df2["Close"].values.reshape(-1, 1)
    y = df2["Future"].values.reshape(-1, 1)
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression(n_jobs=-1).fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    rmse = math.sqrt(mean_squared_error(y_test, preds))

    last = scaler.transform(X[-1].reshape(1, 1))
    tomorrow = float(model.predict(last))
    return tomorrow, rmse


def create_lr_fig(df: pd.DataFrame) -> plt.Figure:
    """
    Returns a Matplotlib Figure of actual vs predicted for the Linear Regression model.
    """
    df2 = df[["Close"]].copy()
    df2["Future"] = df2["Close"].shift(-7)
    df2.dropna(inplace=True)

    X = df2["Close"].values.reshape(-1, 1)
    y = df2["Future"].values.reshape(-1, 1)
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression(n_jobs=-1).fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(y_test, label="Actual Price")
    ax.plot(preds, label="Predicted Price")
    ax.set_title("Linear Regression Fit")
    ax.legend()
    return fig


# ─── 5. Sentiment stub & recommendation ──────────────────────────────────────
# def retrieving_tweets_polarity(symbol: str):
#     """
#     Stub – replace with actual Twitter API logic if desired.
#     """
#     return 0.0, [], "Cannot fetch tweets", 0, 0, 0
def retrieving_tweets_polarity(symbol: str, max_tweets: int = 50):
    """
    Fetch recent tweets for `symbol`, analyze their sentiment via VADER,
    and return:
      - avg compound polarity (float)
      - list of raw tweet texts (List[str])
      - overall label ("Positive"/"Negative"/"Neutral")
      - counts of (pos, neg, neu)
    """
    # 1. Pull the tweets
    tweets = fetch_twitter_texts(symbol, max_tweets=max_tweets)

    # 2. Run VADER sentiment on them
    sent = analyze_sentiments(tweets)

    # 3. Unpack results
    polarity = sent["avg_compound"]
    tw_list  = tweets
    tw_label = sent["label"]
    pos      = sent["counts"]["pos"]
    neg      = sent["counts"]["neg"]
    neu      = sent["counts"]["neu"]

    return polarity, tw_list, tw_label, pos, neg, neu

def recommending(df, polarity, today_close, mean_future):
    today = _scalar(today_close)
    meanf = _scalar(mean_future)
    if today < meanf and polarity > 0:
        return "RISE", "BUY"
    return "FALL", "SELL"


# # ─── 6. Master runner ─────────────────────────────────────────────────────────


def run_prediction(quote: str, alpha_key: str = None) -> dict:
    df = get_historical(quote, fallback_key=alpha_key, csv_dir=".")
    if df.empty:
        return {}

    today = df.iloc[-1]
    arima_tom, arima_rmse = ARIMA_ALGO(df)
    lstm_tom, lstm_rmse, lstm7 = LSTM_ALGO(df)
    lr_tom, lr_rmse = LIN_REG_ALGO(df)
    arima7 = forecast_arima_7(df)

    
    sent = get_multi_social_sentiment(quote)
    overall = sent["overall"]
    idea, decision = make_recommendation(overall["avg_compound"])

    open_v = round(_scalar(today["Open"]), 2)
    high_v = round(_scalar(today["High"]), 2)
    low_v = round(_scalar(today["Low"]), 2)
    close_v = round(_scalar(today["Close"]), 2)
    adj_v = round(_scalar(today.get("Adj Close", today["Close"])), 2)
    vol_v = int(_scalar(today["Volume"]))

    return {
        "latest": {"open": open_v, "high": high_v, "low": low_v,
                   "close": close_v, "adj": adj_v, "vol": vol_v},
        "arima": {"tom": arima_tom, "rmse": arima_rmse},
        "arima7": arima7,
        "lstm7": lstm7,
        "lstm": {"tom": lstm_tom, "rmse": lstm_rmse},
        "lr": {"tom": lr_tom, "rmse": lr_rmse},
        "sent": sent,
        "rec": {"idea": idea, "decision": decision}
    }


def forecast_arima_7(df: pd.DataFrame, order=(6,1,0)) -> list[float]:
    """
    Fit ARIMA on the full series and forecast the next 7 values.
    Returns a plain Python list of floats.
    """
    series = df["Close"]
    model = ARIMA(series, order=order)
    fit   = model.fit()
    fcast = fit.forecast(steps=7)
    # turn numpy floats into Python floats
    return [float(x) for x in fcast]

def _scalar(x):
    """
    Safely convert to float scalar from numpy type or Series.
    """
    if hasattr(x, "item"):
        return x.item()
    return float(x)

def retrieving_tweets_polarity(symbol: str, max_tweets: int = 50):
    """
    Dummy sentiment fallback.
    """
    print(f"[INFO] Skipping Twitter sentiment for {symbol}")
    polarity = 0.0
    tw_list = [f"No Twitter data for {symbol}"]
    tw_label = "Neutral"
    pos = neg = neu = 0
    return polarity, tw_list, tw_label, pos, neg, neu



if __name__ == "__main__":
    df = get_historical("GOOG", fallback_key="JB37LT7KI12GIN2D", csv_dir=".")
    print(df.tail())
