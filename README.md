# 📊 Stock Insights – Smart Market Prediction and Analysis App

## 📘 Project Overview: What is *Stock Insights*?

**Stock Insights** is an intelligent, real-time **stock market analysis and forecasting web application** built with **Flask** and powered by **machine learning** and **natural language processing**.

It’s designed for **investors, analysts, and students** who want a one-stop platform to:

- 📉 **Visualize live stock prices** with interactive Plotly charts
- 📊 **Explore stock performance** and fundamental financial data
- 🧠 **Forecast prices** using LSTM and Prophet models
- 💬 **Analyze sentiment** from Twitter, Reddit, and 4chan
- 📈 **Compare stocks** on volatility, returns, and performance
- 📰 **View live financial news** from trusted sources
- 🚀 **Track real-time market trends** like top gainers, losers, and indices

Think of it as a streamlined, modular **alternative to Bloomberg or TradingView** with AI-enhanced forecasting.

---

## 🚀 Features

- 🔐 Secure user authentication (Signup/Login/Logout)
- 📈 Explore individual stock performance and fundamentals
- 🧠 Stock prediction using LSTM + Prophet
- 📰 Financial news aggregation (Business Today)
- 📊 Real-time stock visualizer (Candlestick, OHLC, Bar, Line charts)
- 📌 Market trends (Top Gainers, Losers, and Indices)
- ⚖️ Stock comparison dashboard (returns, volatility, cumulative)
- 🧠 Sentiment analysis from Twitter, Reddit, and 4chan

---

## 🛠️ Tech Stack

- **Backend:** Flask, SQLite3
- **Frontend:** HTML, CSS, Bootstrap, JavaScript
- **APIs:** Yahoo Finance (yfinance), YahooQuery, YahooFin
- **Visualization:** Plotly, Matplotlib, Folium
- **ML Models:** LSTM (Keras), Prophet, Scikit-learn
- **NLP:** VADER Sentiment
- **Hosting:** Firebase (static), Render/Heroku (dynamic)

---

## 🧩 Folder Structure

    📁 stock_project/
    ├── app.py                  # Main Flask application
    ├── templates/              # HTML pages
    ├── static/                 # Static assets (CSS, JS, media, CSV)
    ├── data/                   # SQLite DB and CSV data files
    ├── routes/                 # Modular Flask routes
    ├── prediction.py           # ML models & forecasting logic
    ├── sentiment.py            # Sentiment analysis engine
    ├── requirements.txt        # All dependencies
    └── README.md               # You're here!

---

## 🧪 Setup Instructions

1. **Clone the repository**
    ```bash
    git clone https://github.com/NagaRamya1531-tech/stock_project.git
    cd stock_project
    ```

2. **Create and activate a virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the app**
    ```bash
    python app.py
    ```

Then visit `http://127.0.0.1:5000/` in your browser.

---

## 📘 Usage Guide

| Feature            | URL                       | Description                              |
|--------------------|---------------------------|------------------------------------------|
| Home               | `/`                       | Welcome page and intro                   |
| Signup/Login       | `/signup`, `/login`       | User authentication                      |
| Dashboard          | `/dashboard`              | Central hub with live data and links     |
| Explore            | `/explore`                | Stock performance + chart + metrics      |
| News               | `/news`                   | Latest business headlines                |
| Market Trends      | `/market_trends`          | Gainers, losers, indices                 |
| Stock Prediction   | `/stock_lstm_prediction`  | LSTM + Prophet + Sentiment forecast      |
| Live Visualizer    | `/live_visualize`         | Realtime chart (candlestick, OHLC, etc.) |
| Compare Stocks     | `/stock_comparison`       | Compare performance and risk             |

---

## 🧠 Model Details

- **LSTM**: Predicts future prices based on 10+ years of closing prices
- **Prophet**: Time series decomposition for trend and seasonality
- **Sentiment**: VADER sentiment scores from Twitter, Reddit, 4chan
- **Recommendation Engine**: Uses aggregated sentiment to advise action

---

## 🌐 Deployment

- **Firebase** for static hosting (only front-end pages)
- **Render/Heroku** for complete backend + UI
- `.env` can be used to secure keys (if extended to production)

---

## 👤 Author

**Naga Ramya Gurrala**  
GitHub: [@NagaRamya1531-tech](https://github.com/NagaRamya1531-tech)

---

## 📄 License

Licensed under the [MIT License](LICENSE).