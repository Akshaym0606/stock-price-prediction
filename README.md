# 📈 AI Stock Price Prediction System

A machine learning project that predicts Apple (AAPL) stock prices by combining
historical stock data, news sentiment analysis, and deep learning models —
displayed in an interactive Streamlit dashboard.

---

## 🎯 Project Overview

This system uses three different approaches to forecast stock prices:
- **LSTM** (Long Short-Term Memory) — Deep learning model
- **Prophet** — Facebook's time series forecasting model
- **ARIMA** — Classical statistical forecasting model

It also incorporates **news sentiment analysis** to understand how financial
news affects stock prices, and displays everything in a live interactive dashboard.

---

## 🏆 Results

| Model   | MAE     | RMSE    | R² Score |
|---------|---------|---------|----------|
| LSTM    | $4.29   | $5.09   | **0.925** ✅ |
| Prophet | $27.25  | $31.39  | -1.58    |
| ARIMA   | $40.90  | $45.33  | -4.38    |

**LSTM achieved 92.5% accuracy** (R² = 0.925) — the best performing model.

---

## 📁 Project Structure

```
stock-price-prediction/
│
├── README.md
├── requirements.txt
│
├── member1_data/
│   ├── stock_data_preprocessing.py   # Downloads & preprocesses AAPL data
│   ├── stock_data_processed.csv      # Cleaned dataset with 23 features
│   └── stock_data_summary.csv        # Statistical summary
│
├── member2_sentiment/
│   ├── news_headlines.csv            # Apple news headlines
│   └── headline_sentiment.csv        # VADER sentiment scores per headline
│
├── member3_models/
│   ├── arima_predictions.csv         # ARIMA model predictions
│   ├── prophet_predictions.csv       # Prophet model predictions
│   ├── lstm_predictions.csv          # LSTM model predictions
│   ├── evaluation_report.csv         # Model comparison metrics
│   ├── arima_model.pkl               # Saved ARIMA model
│   ├── prophet_model.pkl             # Saved Prophet model
│   ├── model_weights.h5              # LSTM model weights
│   ├── config.json                   # LSTM model architecture
│   └── metadata.json                 # Model metadata
│
└── member4_dashboard/
    └── stock_dashboard.py            # Streamlit interactive dashboard
```

---

## 🚀 How to Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/stock-price-prediction.git
cd stock-price-prediction
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Copy CSV files to dashboard folder
Copy all CSV files into the `member4_dashboard/` folder:
```
member4_dashboard/
├── stock_dashboard.py
├── stock_data_processed.csv
├── headline_sentiment.csv
├── lstm_predictions.csv
├── prophet_predictions.csv
├── arima_predictions.csv
└── evaluation_report.csv
```

### Step 4 — Run the dashboard
```bash
cd member4_dashboard
streamlit run stock_dashboard.py
```

### Step 5 — Open in browser
```
http://localhost:8501
```

---

## 📊 Dashboard Features

| Tab | What it shows |
|-----|--------------|
| 📊 Price & Indicators | Live candlestick chart, MA, RSI, MACD, Bollinger Bands |
| 🗞️ Sentiment | Daily news sentiment scores, trend, pie breakdown |
| 🤖 Predictions | LSTM vs Prophet vs ARIMA vs Actual price comparison |
| ⚖️ Backtesting | MA Crossover strategy vs Buy & Hold comparison |
| 📋 Model Evaluation | RMSE, MAE, R² score comparison with charts |

---

## 🛠️ Technologies Used

### Member 1 — Data Collection & Preprocessing
- **yfinance** — Download stock data from Yahoo Finance
- **Pandas / NumPy** — Data cleaning and feature engineering
- **ta (Technical Analysis library)** — RSI, MACD, Bollinger Bands, ATR

### Member 2 — News Sentiment Analysis
- **NewsAPI** — Fetch financial news headlines
- **VADER** — Sentiment analysis on headlines
- **NLTK** — Natural language processing

### Member 3 — Prediction Models
- **TensorFlow / Keras** — LSTM deep learning model
- **Prophet** — Facebook time series forecasting
- **Statsmodels** — ARIMA classical forecasting
- **Scikit-learn** — Model evaluation metrics

### Member 4 — Frontend & Visualization
- **Streamlit** — Web dashboard framework
- **Plotly** — Interactive charts and graphs
- **yfinance** — Live stock data for any ticker

---

## 📈 Dataset

- **Stock:** Apple Inc. (AAPL)
- **Date Range:** 2020-01-01 to 2023-12-31
- **Source:** Yahoo Finance via yfinance
- **Features:** 23 columns including OHLCV, MA, RSI, MACD, Bollinger Bands, ATR, Sentiment

---

## 👥 Team

| Member | Role | Technologies |
|--------|------|-------------|
| Member 1 | Data Collection & Preprocessing | Python, yfinance, Pandas, ta |
| Member 2 | News Sentiment Analysis | NewsAPI, VADER, NLTK |
| Member 3 | Prediction Models | LSTM, Prophet, ARIMA, Keras |
| Member 4 | Frontend & Backtesting | Streamlit, Plotly |

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.
Predictions shown are not financial advice.
Past performance does not guarantee future results.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
