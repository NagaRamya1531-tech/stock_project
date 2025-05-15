import os
import sqlite3
import bcrypt
import json
import requests
import numpy as np
import pandas as pd
# import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
import json
import requests
from geopy.geocoders import Nominatim
import folium
from folium import Map, Marker
from io import BytesIO
import base64
import io
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

from yahoo_fin import stock_info as si
from sklearn.linear_model import LinearRegression
from io import BytesIO
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from prophet import Prophet 
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.io as pio

import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from sklearn.cluster import KMeans

pio.renderers.default = "browser" 

# app.py

from sentiment import (
    fetch_twitter_texts, fetch_reddit_texts, fetch_4chan_texts,
    analyze_sentiments, plot_sentiment_pie, get_multi_social_sentiment,
    make_recommendation
)
from prediction import (
    get_historical,
    run_prediction,
    create_trend_fig,
    create_arima_fig,
    create_lstm_fig,
    create_lr_fig
)


from io import BytesIO



YAHOO_SCREENER_PREDEF = (
    "https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved"
)

YAHOO_SCREENER_URL = "https://query2.finance.yahoo.com/v1/finance/screener"
# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Database setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, 'data', 'database.db')

def init_db():
    os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            dob TEXT NOT NULL,
            email TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# --------------------------- Auth Routes ---------------------------

@app.route('/')
def home():
    return render_template('home.html')




@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        dob = request.form['dob']
        username = request.form['username']
        password = request.form['password']
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        try:
            conn = sqlite3.connect(DATABASE)
            c = conn.cursor()
            c.execute('INSERT INTO users (name, dob, email, username, password) VALUES (?, ?, ?, ?, ?)',
                      (name, dob, email, username, hashed_pw))
            conn.commit()
            flash('Signup successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists.', 'danger')
        finally:
            conn.close()
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_input = request.form['username']  # Could be username or email
        password = request.form['password']

        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('SELECT username, password FROM users WHERE username = ? OR email = ?', (user_input, user_input))
        result = c.fetchone()
        conn.close()

        if result and bcrypt.checkpw(password.encode('utf-8'), result[1]):
            session['username'] = result[0]  # Store actual username in session
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username/email or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))



@app.route('/dashboard')
def dashboard():
    import pandas as pd

    df = pd.read_csv('static/data/Yahoo-Finance-Ticker-Symbols.csv')
    df = df.dropna(subset=["Ticker", "Name", "Exchange"])

    exchange_map = {
        "NMS": "NASDAQ", "NYQ": "NYSE", "ASE": "AMEX", "PCX": "NYSEARCA",
        "NGM": "NASDAQ", "PNK": "OTC", "BTS": "BATS"
    }
    df["TVExchange"] = df["Exchange"].map(exchange_map)
    df = df.dropna(subset=["TVExchange"])

    tv_symbols = [
        {"s": f"{row.TVExchange}:{row.Ticker}", "d": row.Name}
        for _, row in df.head(30).iterrows()
    ]

    return render_template("dashboard.html", tv_symbols=tv_symbols)



@app.route('/get_ticker_symbols')
def get_ticker_symbols():
    try:
        df = pd.read_csv('static/data/Yahoo-Finance-Corrected.csv')

        # Ensure these columns exist
        if not {'Symbol', 'Name', 'Exchange'}.issubset(df.columns):
            return jsonify({"error": "CSV must contain 'Symbol', 'Name', 'Exchange' columns"}), 400

        df = df.dropna(subset=["Symbol", "Name", "Exchange"])
        df = df[df["Exchange"].isin(["NASDAQ", "NYSE"])]
        df = df.head(50)  # Limit to avoid overload

        symbols = []
        for _, row in df.iterrows():
            symbol = str(row['Symbol']).strip().upper()
            name = str(row['Name']).strip()
            exchange = str(row['Exchange']).strip().upper()

            prefix = "NASDAQ" if exchange == "NASDAQ" else "NYSE"
            symbols.append({
                "proName": f"{prefix}:{symbol}",
                "title": name
            })

        return jsonify(symbols)

    except Exception as e:
        import traceback
        print("ERROR:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500



@app.route("/debug_columns")
def debug_columns():
    import pandas as pd
    df = pd.read_csv("static/data/Yahoo-Finance-Ticker-Symbols_test.csv")
    return f"Columns: {list(df.columns)}"



# --------------------------- Company Page ---------------------------

@app.route('/company', methods=['GET', 'POST'])
def company():
    # load & clean
    data_dir = os.path.join('static', 'data')
    os.makedirs(data_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(data_dir, 'fdata.csv'), encoding='latin-1')
    df['Industry'] = df['Industry'].fillna('Unknown')

    industries = sorted(df['Industry'].unique())
    selected_industry = request.form.get('industry') if request.method == 'POST' else None

    # create a list of simple dicts
    companies = []
    if selected_industry:
        sub = df[df['Industry'] == selected_industry][['Company Name','Symbol','Market Cap']]
        companies = sub.to_dict(orient='records')

    return render_template(
        'company.html',
        industries=industries,
        selected_industry=selected_industry,
        companies=companies
    )


# --------------------------- Explore ---------------------------


from flask import render_template, request
import pandas as pd
import requests
import plotly.graph_objs as go

ALPHA_API_KEY = '723Z6Q31E5B0F2W3'

@app.route('/explore', methods=['GET', 'POST'])
def explore():
    chart_html = ""
    data = {}
    ticker = "AAPL"
    selected_time_range = "1mo"

    # Time range mapping for Alpha Vantage
    time_range_map = {
        "1d": ("TIME_SERIES_INTRADAY", "1min"),
        "5d": ("TIME_SERIES_INTRADAY", "5min"),
        "1mo": ("TIME_SERIES_DAILY", None),
        "6mo": ("TIME_SERIES_DAILY", None),
        "1y": ("TIME_SERIES_DAILY", None),
        "5y": ("TIME_SERIES_WEEKLY", None),
        "all": ("TIME_SERIES_MONTHLY", None)
    }

    if request.method == "POST":
        ticker = request.form.get("ticker", "AAPL").upper()
        selected_time_range = request.form.get("time_range", "1mo")

    function, interval = time_range_map.get(selected_time_range, ("TIME_SERIES_DAILY", None))
    url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={ALPHA_API_KEY}"
    if interval:
        url += f"&interval={interval}&outputsize=compact"

    try:
        # Fetch data from Alpha Vantage
        r = requests.get(url)
        raw = r.json()

        if "Note" in raw or "Error Message" in raw:
            raise ValueError(f"Alpha Vantage Error: {raw.get('Note') or raw.get('Error Message')}")

        key = next((k for k in raw if "Time Series" in k), None)
        if not key:
            raise ValueError("No price data found in Alpha Vantage response.")

        df = pd.DataFrame.from_dict(raw[key], orient="index")
        df.columns = [c.split(". ")[1] for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.astype(float).reset_index()
        df.rename(columns={"index": "Date"}, inplace=True)

        # Plot chart
        fig = go.Figure(go.Scatter(
            x=df["Date"],
            y=df["close"],
            mode="lines+markers",
            name=f"{ticker} Close"
        ))
        fig.update_layout(
            title=f"{ticker} Price Over Time",
            xaxis_title="Date",
            yaxis_title="Close Price",
            template="plotly_white"
        )
        chart_html = fig.to_html(full_html=False)

        # Performance metrics
        pct_returns = df["close"].pct_change().dropna()
        stdev = round(pct_returns.std() * 100, 2)
        annual_return = round((1 + pct_returns.mean()) ** 252 - 1, 2) * 100
        risk_adj_return = round(annual_return / stdev, 2) if stdev != 0 else 0

        latest_price = round(df["close"].iloc[-1], 2)
        previous_close = round(df["close"].iloc[-2], 2)
        price_diff = round(latest_price - previous_close, 2)
        percent_change = round((price_diff / previous_close) * 100, 2)
        color = "success" if price_diff > 0 else "danger"

        data = {
            "ticker": ticker,
            "latest_price": latest_price,
            "price_diff": price_diff,
            "percent_change": percent_change,
            "color": color,
            "annual_return": annual_return,
            "stdev": stdev,
            "risk_adj_return": risk_adj_return,
            "fundamentals": {},  # Empty dict to support template
            "location": None
        }

    except Exception as e:
        chart_html = f"<p class='text-danger'>Error: {e}</p>"

    return render_template("explore.html", chart_html=chart_html, data=data, ticker=ticker, selected_time_range=selected_time_range)


# --------------------------- News ---------------------------

@app.route('/news', methods=['GET', 'POST'])
def news():
    urls = {
        "Union Budget": "https://www.businesstoday.in/union-budget",
        "Stocks": "https://www.businesstoday.in/markets/stocks",
        "Investment": "https://www.businesstoday.in/personal-finance/investment"
    }
    selected_category = request.form.get('category', 'Stocks') if request.method == 'POST' else 'Stocks'
    headlines = []

    try:
        response = requests.get(urls[selected_category], headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.content, "html.parser")
        for link in soup.find_all('a'):
            if link.string and len(link.string.strip()) > 35:
                headlines.append(link.string.strip())
    except Exception as e:
        flash(f"Error fetching news: {e}", "danger")

    return render_template('news.html', headlines=headlines, selected_category=selected_category)

# --------------------------- Market Trends ---------------------------

def get_tickers_from_names(companies):
    tickers = {}
    for company in companies:
        try:
            results = search(company).get('quotes', [])
            tickers[company] = results[0]['symbol'] if results else None
        except:
            tickers[company] = None
    return tickers



def fetch_stock_data(ticker, period="1d", interval="1m"):
    try:
        t = Ticker(ticker)
        data = t.history(period=period, interval=interval)
        
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index()
            data = data[data['symbol'] == ticker]
        
        return data
    except Exception as e:
        print(f"[ERROR] Fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_indices():
    indices = {
        "Nifty 50": "^NSEI", "Nifty Bank": "^NSEBANK", "Sensex": "^BSESN",
        "Finnifty": "NIFTY_FIN_SERVICE.NS", "Nifty 100": "^CNX100",
        "S&P 500": "^GSPC", "Dow Jones": "^DJI"
    }
    result = {}
    for name, ticker in indices.items():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if len(hist) >= 2:
                prev = hist['Close'].iloc[-2]
                curr = hist['Close'].iloc[-1]
                change = curr - prev
                percent = (change / prev) * 100
                result[name] = {'close': curr, 'change': change, 'percent': percent}
        except:
            result[name] = {'close': None, 'change': None, 'percent': None}
    return result


# --------------------------- Live Visualize ---------------------------


ALPHA_API_KEY = '723Z6Q31E5B0F2W3'  # Replace with your own if needed

@app.route('/live_visualize', methods=["GET", "POST"])
def live_visualize():
    chart_html, df_table = "", ""
    default_ticker = "SPY"
    chart_type = request.form.get("chart_type", "Line Chart")

    if request.method == "POST":
        ticker = request.form.get("ticker", default_ticker).upper()

        try:
            # Fetch from Alpha Vantage
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval=1min&apikey={ALPHA_API_KEY}'
            r = requests.get(url)
            raw = r.json()

            time_series = raw.get("Time Series (1min)", {})
            if not time_series:
                raise ValueError("No data returned or API limit exceeded.")

            # Parse and clean data
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df = df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            })
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.astype(float)
            df.reset_index(inplace=True)
            df.rename(columns={"index": "timestamp"}, inplace=True)

            # Generate chart
            if chart_type == "Line Chart":
                fig = go.Figure(go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name='Close Price'))
            elif chart_type == "Candlestick Chart":
                fig = go.Figure(go.Candlestick(x=df['timestamp'],
                                               open=df['open'],
                                               high=df['high'],
                                               low=df['low'],
                                               close=df['close']))
            elif chart_type == "Bar Chart":
                fig = go.Figure(go.Bar(x=df['timestamp'], y=df['close'], name='Close Price'))
            elif chart_type == "OHLC Chart":
                fig = go.Figure(go.Ohlc(x=df['timestamp'],
                                        open=df['open'],
                                        high=df['high'],
                                        low=df['low'],
                                        close=df['close']))
            else:
                fig = go.Figure()

            fig.update_layout(
                title=f"{ticker} - {chart_type}",
                xaxis_title="Time",
                yaxis_title="Price",
                plot_bgcolor="white"
            )

            chart_html = fig.to_html(full_html=False)
            df_table = df.tail(20).to_html(classes="table table-striped", border=0, index=False)

        except Exception as e:
            chart_html = f"<p class='text-danger'>Error: {e}</p>"

    return render_template("live_visualize.html", chart_html=chart_html, df_table=df_table)



# --------------------------- Stock Prediction ---------------------------

ALPHA_KEY = "723Z6Q31E5B0F2W3"  # <-- Replace with your actual API key

@app.route('/stock_comparison', methods=['GET', 'POST'])
def stock_comparison():
    try:
        df = pd.read_csv(os.path.join('static', 'data', 'fdata.csv'), encoding='latin-1')
        tickers = sorted(df['Symbol'].dropna().unique().tolist())
    except Exception:
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

    ticker1 = ticker2 = ""
    fig_returns = fig_cumulative = fig_volatility = ""
    error = None

    if request.method == 'POST':
        ticker1 = request.form.get('ticker1')
        ticker2 = request.form.get('ticker2')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')

        ts = TimeSeries(key=ALPHA_KEY, output_format='pandas')

        try:
            data1, _ = ts.get_daily(symbol=ticker1, outputsize='full')
            data2, _ = ts.get_daily(symbol=ticker2, outputsize='full')


            data1 = data1.sort_index()
            data2 = data2.sort_index()

            # Filter by date
            data1 = data1.loc[start_date:end_date]
            data2 = data2.loc[start_date:end_date]

            if data1.empty or data2.empty:
                error = "❌ One or both tickers returned no data."
            else:
                # Rename '5. adjusted close' to Close
                # Rename '4. close' to Close
                data1.rename(columns={'4. close': 'Close'}, inplace=True)
                data2.rename(columns={'4. close': 'Close'}, inplace=True)


                # Daily returns
                data1['Daily_Return'] = data1['Close'].pct_change()
                data2['Daily_Return'] = data2['Close'].pct_change()

                # Cumulative returns
                data1['Cumulative_Return'] = (1 + data1['Daily_Return']).cumprod() - 1
                data2['Cumulative_Return'] = (1 + data2['Daily_Return']).cumprod() - 1

                # Volatility
                vol1 = data1['Daily_Return'].std()
                vol2 = data2['Daily_Return'].std()

                # Daily Return Chart
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=data1.index, y=data1['Daily_Return'], mode='lines', name=ticker1))
                fig1.add_trace(go.Scatter(x=data2.index, y=data2['Daily_Return'], mode='lines', name=ticker2))
                fig1.update_layout(title='Daily Returns', xaxis_title='Date', yaxis_title='Return')
                fig_returns = fig1.to_html(full_html=False)

                # Cumulative Return Chart
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=data1.index, y=data1['Cumulative_Return'], mode='lines', name=ticker1))
                fig2.add_trace(go.Scatter(x=data2.index, y=data2['Cumulative_Return'], mode='lines', name=ticker2))
                fig2.update_layout(title='Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Return')
                fig_cumulative = fig2.to_html(full_html=False)

                # Volatility Chart
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(x=[ticker1], y=[vol1], name=ticker1))
                fig3.add_trace(go.Bar(x=[ticker2], y=[vol2], name=ticker2))
                fig3.update_layout(title='Volatility Comparison', xaxis_title='Stock', yaxis_title='Volatility')
                fig_volatility = fig3.to_html(full_html=False)

        except Exception as e:
            error = f"Error fetching data: {e}"

    return render_template("stock_comparison.html",
                           tickers=tickers,
                           ticker1=ticker1,
                           ticker2=ticker2,
                           fig_returns=fig_returns,
                           fig_cumulative=fig_cumulative,
                           fig_volatility=fig_volatility,
                           error=error)






def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')





@app.route('/stock_lstm_prediction', methods=['GET', 'POST'])
def stock_lstm_prediction():
    # Initialize variables
    symbol = None
    latest = arima = lstm = lr = sent = rec = {}
    charts = {}
    lstm7 = []
    rec_text = error = None

    # ✅ Default `social` structure to prevent template errors on GET or failure
    social = {
        'twitter': {"label": "Neutral", "avg_compound": 0.0, "counts": {"pos": 0, "neg": 0, "neu": 0}},
        'reddit':  {"label": "Neutral", "avg_compound": 0.0, "counts": {"pos": 0, "neg": 0, "neu": 0}},
        '4chan':   {"label": "Neutral", "avg_compound": 0.0, "counts": {"pos": 0, "neg": 0, "neu": 0}},
        'overall': {"label": "Neutral", "avg_compound": 0.0, "counts": {"pos": 0, "neg": 0, "neu": 0}},
    }

    twitter_pie_b64 = reddit_pie_b64 = chan_pie_b64 = None

    if request.method == 'POST':
        symbol = request.form['symbol'].strip().upper()

        # Step 1: Fetch historical OHLCV data
        try:
            df = get_historical(symbol, fallback_key="723Z6Q31E5B0F2W3", csv_dir=".")
        except Exception as e:
            df = pd.DataFrame()
            error = f"Error retrieving data for {symbol}: {e}"

        if df.empty:
            error = f"No data found for {symbol}"
        else:
            try:
                # Step 2: Run all prediction models
                res = run_prediction(symbol, alpha_key="723Z6Q31E5B0F2W3")
                latest = res.get('latest', {})
                arima  = res.get('arima', {})
                lstm   = res.get('lstm', {})
                lr     = res.get('lr', {})
                sent   = res.get('sent', {})
                rec    = res.get('rec', {})
                lstm7  = res.get('lstm7', [])

                # Step 3: Get sentiment scores
                social = get_multi_social_sentiment(symbol)
                overall = social.get('overall', {})
                trend, action = make_recommendation(overall.get('avg_compound', 0.0))
                rec_text = (
                    f"According to the Sentiment Analysis of Twitter, Reddit & 4chan, "
                    f"a {trend.upper()} in {symbol} stock is expected ⇒ {action.upper()}"
                )

                # Step 4: Generate model charts
                charts = {
                    'trend': fig_to_base64(create_trend_fig(df)),
                    'arima': fig_to_base64(create_arima_fig(df)),
                    'lstm':  fig_to_base64(create_lstm_fig(df)),
                    'lr':    fig_to_base64(create_lr_fig(df)),
                }

                # Step 5: Sentiment pie charts
                twitter_pie_b64 = fig_to_base64(plot_sentiment_pie(social['twitter'], 'Twitter'))
                reddit_pie_b64  = fig_to_base64(plot_sentiment_pie(social['reddit'], 'Reddit'))
                chan_pie_b64    = fig_to_base64(plot_sentiment_pie(social['4chan'], '4chan'))

            except Exception as e:
                error = f"Model processing error for {symbol}: {e}"

    # Step 6: Render the template
    return render_template(
        'stock_lstm_prediction.html',
        quote=symbol,
        latest=latest,
        arima=arima,
        lstm=lstm,
        lr=lr,
        sent=sent,
        rec=rec,
        charts=charts,
        lstm7=lstm7,
        social=social,
        twitter_pie=twitter_pie_b64,
        reddit_pie=reddit_pie_b64,
        chan_pie=chan_pie_b64,
        rec_text=rec_text,
        error=error
    )


# --------------------------- Run Server ---------------------------

if __name__ == '__main__':
    init_db()

