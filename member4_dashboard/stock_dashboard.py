import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock AI Dashboard", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0b0e1a; color: #e6edf3; }
section[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #21262d; }
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #161b27 0%, #1c2333 100%);
    border: 1px solid #30363d; border-radius: 14px; padding: 18px 20px;
}
div[data-testid="metric-container"] label { color: #8b949e !important; font-size: 12px; text-transform: uppercase; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 26px; font-weight: 700; }
.page-title { font-size: 32px; font-weight: 800; background: linear-gradient(90deg, #58a6ff, #a371f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.section-title { font-size: 14px; font-weight: 600; color: #8b949e; margin: 20px 0 10px 0;
    padding-bottom: 8px; border-bottom: 1px solid #21262d; text-transform: uppercase; letter-spacing: 0.08em; }
.signal-buy  { background:#0d3321; color:#3fb950; padding:8px 20px; border-radius:24px; font-weight:800; font-size:18px; display:inline-block; border:1px solid #3fb950; }
.signal-sell { background:#3d1218; color:#f85149; padding:8px 20px; border-radius:24px; font-weight:800; font-size:18px; display:inline-block; border:1px solid #f85149; }
.signal-hold { background:#2d2208; color:#e3b341; padding:8px 20px; border-radius:24px; font-weight:800; font-size:18px; display:inline-block; border:1px solid #e3b341; }
.info-box { background: #161b27; border: 1px solid #30363d; border-radius: 10px; padding: 14px 18px; margin: 8px 0; font-size: 14px; color: #c9d1d9; }
.real-badge { background:#0d3321; color:#3fb950; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; }
.dummy-badge { background:#2d2208; color:#e3b341; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

CARD  = '#161b27'
GRID  = '#21262d'
GREEN = '#3fb950'
RED   = '#f85149'
BLUE  = '#58a6ff'
GOLD  = '#e3b341'
PURP  = '#a371f7'

CHART_BASE = dict(
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    font=dict(color='#c9d1d9', size=12),
    yaxis=dict(gridcolor=GRID, showgrid=True, zeroline=False),
    legend=dict(bgcolor=CARD, bordercolor=GRID, borderwidth=1),
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode='x unified'
)

# ── DATA LOADING ───────────────────────────────────────────────────────────────
@st.cache_data
def load_stock_data(days):
    try:
        df = pd.read_csv('stock_data_processed.csv', index_col='Date', parse_dates=True)
        df = df.sort_index()
        df = df.tail(days).copy()
        # Standardise column names
        df.rename(columns={'MA_10':'MA10','MA_20':'MA20','MA_50':'MA50',
                            'MACD_Hist':'MACD_Hist','BB_Middle':'BB_Mid'}, inplace=True)
        if 'MACD_Hist' not in df.columns and 'MACD' in df.columns:
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        if 'BB_Mid' not in df.columns:
            df['BB_Mid'] = df['Close'].rolling(20).mean()
        if 'MA10' not in df.columns:
            df['MA10'] = df['Close'].rolling(10).mean()
        df = df.reset_index()
        df.rename(columns={'index':'Date'}, inplace=True)
        return df, True
    except FileNotFoundError:
        return None, False

@st.cache_data
def load_sentiment_data():
    try:
        df = pd.read_csv('headline_sentiment.csv', parse_dates=['date'])
        df.rename(columns={'date':'Date'}, inplace=True)
        daily = df.groupby('Date')['sentiment'].mean().reset_index()
        daily.rename(columns={'sentiment':'Sentiment'}, inplace=True)
        return daily, True
    except FileNotFoundError:
        return None, False

@st.cache_data
def load_predictions():
    try:
        lstm    = pd.read_csv('lstm_predictions.csv',    parse_dates=['Date'])
        prophet = pd.read_csv('prophet_predictions.csv', parse_dates=['Date'])
        arima   = pd.read_csv('arima_predictions.csv',   parse_dates=['Date'])
        lstm.rename(columns={'Predicted':'LSTM_Pred'}, inplace=True)
        prophet.rename(columns={'Predicted':'Prophet_Pred','Actual':'Actual_P'}, inplace=True)
        arima.rename(columns={'Predicted':'ARIMA_Pred','Actual':'Actual_A'}, inplace=True)
        df = lstm.merge(prophet[['Date','Prophet_Pred','Actual_P']], on='Date', how='outer')
        df = df.merge(arima[['Date','ARIMA_Pred']], on='Date', how='outer')
        df['Actual'] = df['Actual'].combine_first(df['Actual_P'])
        df.drop(columns=['Actual_P'], inplace=True)
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df, True
    except FileNotFoundError:
        return None, False

@st.cache_data
def load_evaluation():
    try:
        df = pd.read_csv('evaluation_report.csv')
        return df, True
    except FileNotFoundError:
        return None, False

@st.cache_data
def load_future_forecast():
    """Loads the 30-day forward forecast file (LSTM/Prophet/ARIMA) if present."""
    try:
        df = pd.read_csv('future_forecast.csv', parse_dates=['Date'])
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df, True
    except FileNotFoundError:
        return None, False
    except Exception:
        return None, False

def find_col(df, keyword):
    """Finds a column whose name contains the given keyword (case-insensitive)."""
    if df is None:
        return None
    for c in df.columns:
        if keyword.lower() in c.lower():
            return c
    return None

# ── LOAD ALL DATA ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Stock AI")
    st.markdown("---")
    st.markdown("### 🎯 Stock")
    ticker = st.selectbox("Stock", ["AAPL"], label_visibility="collapsed")
    st.markdown("### 📅 Time Range")
    rng  = st.radio("Range", ["1M","3M","6M","1Y","All"], horizontal=True, label_visibility="collapsed")
    days = {"1M":22,"3M":66,"6M":132,"1Y":252,"All":9999}[rng]
    st.markdown("### 📊 Indicators")
    show_ma  = st.checkbox("Moving Averages", value=True)
    show_bb  = st.checkbox("Bollinger Bands", value=False)
    show_vol = st.checkbox("Volume", value=True)
    st.markdown("---")
    st.markdown("### 🔌 Data Status")

df_stock, stock_ok   = load_stock_data(days)
df_sent,  sent_ok    = load_sentiment_data()
df_pred,  pred_ok    = load_predictions()
df_eval,  eval_ok    = load_evaluation()
df_future, future_ok = load_future_forecast()

with st.sidebar:
    st.markdown(f'<span class="{"real-badge" if stock_ok else "dummy-badge"}">{"✅ Real" if stock_ok else "⚠️ Dummy"}</span> Stock Data', unsafe_allow_html=True)
    st.markdown(f'<span class="{"real-badge" if sent_ok else "dummy-badge"}">{"✅ Real" if sent_ok else "⚠️ Dummy"}</span> Sentiment', unsafe_allow_html=True)
    st.markdown(f'<span class="{"real-badge" if pred_ok else "dummy-badge"}">{"✅ Real" if pred_ok else "⚠️ Dummy"}</span> Predictions', unsafe_allow_html=True)
    st.markdown(f'<span class="{"real-badge" if eval_ok else "dummy-badge"}">{"✅ Real" if eval_ok else "⚠️ Dummy"}</span> Evaluation', unsafe_allow_html=True)
    st.markdown(f'<span class="{"real-badge" if future_ok else "dummy-badge"}">{"✅ Real" if future_ok else "⚠️ Live Estimate"}</span> Future Forecast', unsafe_allow_html=True)
    st.caption("Place CSV files in same folder as stock_dashboard.py")

# ── FALLBACK DUMMY DATA if CSVs not found ─────────────────────────────────────
if df_stock is None:
    np.random.seed(42)
    n = min(days, 252)
    dates = pd.date_range(end=datetime.today(), periods=n, freq='B')
    price = 150.0; prices = []
    for _ in range(n):
        price *= (1 + np.random.normal(0.0005, 0.016)); prices.append(price)
    df_stock = pd.DataFrame({'Date':dates,'Close':prices})
    df_stock['Open']  = df_stock['Close'].shift(1).fillna(df_stock['Close'])
    df_stock['High']  = df_stock['Close'] * 1.01
    df_stock['Low']   = df_stock['Close'] * 0.99
    df_stock['Volume']= np.random.randint(4_000_000,35_000_000,n)
    df_stock['MA10']  = df_stock['Close'].rolling(10).mean()
    df_stock['MA20']  = df_stock['Close'].rolling(20).mean()
    df_stock['MA50']  = df_stock['Close'].rolling(50).mean()
    df_stock['BB_Mid']  = df_stock['Close'].rolling(20).mean()
    df_stock['BB_Upper']= df_stock['BB_Mid'] + 2*df_stock['Close'].rolling(20).std()
    df_stock['BB_Lower']= df_stock['BB_Mid'] - 2*df_stock['Close'].rolling(20).std()
    delta = df_stock['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df_stock['RSI'] = 100-(100/(1+gain/loss))
    ema12 = df_stock['Close'].ewm(span=12).mean()
    ema26 = df_stock['Close'].ewm(span=26).mean()
    df_stock['MACD']        = ema12-ema26
    df_stock['MACD_Signal'] = df_stock['MACD'].ewm(span=9).mean()
    df_stock['MACD_Hist']   = df_stock['MACD']-df_stock['MACD_Signal']

if df_sent is None:
    np.random.seed(7)
    n2 = len(df_stock)
    df_sent = pd.DataFrame({'Date':df_stock['Date'],'Sentiment':np.clip(np.random.normal(0.05,0.25,n2),-1,1)})

if df_pred is None:
    actual = df_stock['Close'].values[-100:]
    pred_dates = df_stock['Date'].values[-100:]
    np.random.seed(42)
    df_pred = pd.DataFrame({
        'Date': pred_dates,
        'Actual': actual,
        'LSTM_Pred':    actual + np.random.normal(0,5,100),
        'Prophet_Pred': actual + np.random.normal(0,31,100),
        'ARIMA_Pred':   np.full(100, actual.mean()) + np.random.normal(0,20,100),
    })

if df_eval is None:
    df_eval = pd.DataFrame({
        'Model':['ARIMA','Prophet','LSTM'],
        'MAE':[40.90,27.25,4.29],
        'RMSE':[45.33,31.39,5.09],
        'R2 Score':[-4.38,-1.58,0.925]
    })

# ── FUTURE FORECAST FALLBACK (live trend-based estimate) ───────────────────────
# If Member 3's future_forecast.csv isn't in the folder, we build a live
# 30-day forward estimate from the actual stock trend + each model's known
# error (RMSE) so the shape still reflects "LSTM tightest, ARIMA widest".
if df_future is None:
    last_date  = df_stock['Date'].iloc[-1]
    last_price = df_stock['Close'].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)

    recent = df_stock['Close'].tail(20).values
    trend_per_day = (recent[-1] - recent[0]) / max(len(recent)-1, 1)

    def rmse_of(model_name, default):
        m = df_eval.loc[df_eval['Model'] == model_name, 'RMSE']
        return float(m.values[0]) if len(m) else default

    lstm_rmse    = rmse_of('LSTM', 5)
    prophet_rmse = rmse_of('Prophet', 30)
    arima_rmse   = rmse_of('ARIMA', 45)

    np.random.seed(123)
    base_path = last_price + trend_per_day * np.arange(1, 31)

    df_future = pd.DataFrame({
        'Date': future_dates,
        'LSTM_Future':    base_path + np.cumsum(np.random.normal(0, lstm_rmse*0.08, 30)),
        'Prophet_Future': base_path + np.cumsum(np.random.normal(0, prophet_rmse*0.08, 30)),
        'ARIMA_Future':   np.full(30, last_price) + np.cumsum(np.random.normal(0, arima_rmse*0.08, 30)),
    })

lstm_fcol    = find_col(df_future, 'lstm')
prophet_fcol = find_col(df_future, 'prophet')
arima_fcol   = find_col(df_future, 'arima')

# ── KPIs ──────────────────────────────────────────────────────────────────────
cur   = df_stock['Close'].iloc[-1]
prev  = df_stock['Close'].iloc[-2]
chgp  = (cur-prev)/prev*100
rsi   = df_stock['RSI'].iloc[-1] if 'RSI' in df_stock.columns else 50
sent  = df_sent['Sentiment'].iloc[-1]
vol   = df_stock['Volume'].iloc[-1] if 'Volume' in df_stock.columns else 0
h52   = df_stock['High'].max() if 'High' in df_stock.columns else cur*1.1
l52   = df_stock['Low'].min()  if 'Low'  in df_stock.columns else cur*0.9

# Best model accuracy from eval report
best_r2   = df_eval.loc[df_eval['R2 Score'].idxmax(), 'R2 Score']
best_model= df_eval.loc[df_eval['R2 Score'].idxmax(), 'Model']
best_acc  = best_r2 * 100

if sent > 0.2 and rsi < 65:
    signal, sig_cls = "🟢 BUY",  "signal-buy"
elif rsi > 70 or sent < -0.3:
    signal, sig_cls = "🔴 SELL", "signal-sell"
else:
    signal, sig_cls = "🟡 HOLD", "signal-hold"

# ── HEADER ────────────────────────────────────────────────────────────────────
c1,c2 = st.columns([3,1])
with c1:
    st.markdown(f'<div class="page-title">📈 {ticker} — AI Stock Dashboard</div>', unsafe_allow_html=True)
    st.caption(f"🕐 {datetime.now().strftime('%d %B %Y, %I:%M %p')}  •  Best Model: {best_model}  •  Range: {rng}")
with c2:
    st.markdown(f"<br><span class='{sig_cls}'>{signal}</span>", unsafe_allow_html=True)
st.markdown("---")

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Price",          f"${cur:.2f}",      f"{chgp:+.2f}%")
k2.metric("RSI (14)",       f"{rsi:.1f}",        "Overbought" if rsi>70 else ("Oversold" if rsi<30 else "Normal"))
k3.metric("Sentiment",      f"{sent:+.2f}",      "Bullish 🐂" if sent>0 else "Bearish 🐻")
k4.metric("Volume",         f"{vol/1e6:.1f}M")
k5.metric(f"{best_model} Accuracy", f"{best_acc:.1f}%")
k6.metric("52W High/Low",   f"${h52:.0f} / ${l52:.0f}")
st.markdown("")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs(["📊 Price & Indicators","🗞️ Sentiment","🤖 Predictions","⚖️ Backtesting","📋 Model Evaluation"])

# ════════ TAB 1 — PRICE ════════
with tab1:
    rows        = 3 if show_vol else 2
    row_heights = [0.6,0.2,0.2] if show_vol else [0.65,0.35]
    rsi_row     = 3 if show_vol else 2
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        row_heights=row_heights, vertical_spacing=0.04)

    if 'Open' in df_stock.columns:
        fig.add_trace(go.Candlestick(x=df_stock['Date'],
            open=df_stock['Open'],high=df_stock['High'],
            low=df_stock['Low'],close=df_stock['Close'],
            name="OHLC",increasing_line_color=GREEN,decreasing_line_color=RED,
            increasing_fillcolor=GREEN,decreasing_fillcolor=RED), row=1,col=1)
    else:
        fig.add_trace(go.Scatter(x=df_stock['Date'],y=df_stock['Close'],
            name="Close",line=dict(color=BLUE,width=2)), row=1,col=1)

    if show_ma and 'MA10' in df_stock.columns:
        fig.add_trace(go.Scatter(x=df_stock['Date'],y=df_stock['MA10'],name="MA10",line=dict(color='#f0883e',width=1.2)),row=1,col=1)
        fig.add_trace(go.Scatter(x=df_stock['Date'],y=df_stock['MA20'],name="MA20",line=dict(color=BLUE,width=1.5)),row=1,col=1)
        fig.add_trace(go.Scatter(x=df_stock['Date'],y=df_stock['MA50'],name="MA50",line=dict(color=GOLD,width=1.5)),row=1,col=1)

    if show_bb and 'BB_Upper' in df_stock.columns:
        fig.add_trace(go.Scatter(x=df_stock['Date'],y=df_stock['BB_Upper'],name="BB Upper",
            line=dict(color=PURP,width=1,dash='dot')),row=1,col=1)
        fig.add_trace(go.Scatter(x=df_stock['Date'],y=df_stock['BB_Lower'],name="BB Lower",
            line=dict(color=PURP,width=1,dash='dot'),
            fill='tonexty',fillcolor='rgba(163,113,247,0.08)'),row=1,col=1)

    if show_vol and 'Volume' in df_stock.columns:
        if 'Open' in df_stock.columns:
            vc = [GREEN if df_stock['Close'].iloc[i] >= df_stock['Open'].iloc[i] else RED for i in range(len(df_stock))]
        else:
            vc = [GREEN for _ in range(len(df_stock))]
        fig.add_trace(go.Bar(x=df_stock['Date'],y=df_stock['Volume'],
            marker_color=vc,name="Volume",opacity=0.7),row=2,col=1)

    if 'RSI' in df_stock.columns:
        fig.add_trace(go.Scatter(x=df_stock['Date'],y=df_stock['RSI'],
            line=dict(color=PURP,width=2),name="RSI"),row=rsi_row,col=1)
        fig.add_hline(y=70,line_color=RED,line_dash='dash',row=rsi_row,col=1)
        fig.add_hline(y=30,line_color=GREEN,line_dash='dash',row=rsi_row,col=1)

    fig.update_layout(**CHART_BASE,height=600,showlegend=True)
    fig.update_xaxes(gridcolor=GRID,rangeslider_visible=False)
    fig.update_yaxes(gridcolor=GRID,zeroline=False)
    if show_vol:
        fig.update_yaxes(title_text="Price ($)",row=1,col=1)
        fig.update_yaxes(title_text="Volume",row=2,col=1)
        fig.update_yaxes(title_text="RSI",range=[0,100],row=3,col=1)
    else:
        fig.update_yaxes(title_text="Price ($)",row=1,col=1)
        fig.update_yaxes(title_text="RSI",range=[0,100],row=2,col=1)
    st.plotly_chart(fig,use_container_width=True)

    if 'MACD' in df_stock.columns:
        st.markdown('<div class="section-title">MACD</div>',unsafe_allow_html=True)
        fig_macd = go.Figure()
        mc = [GREEN if v>=0 else RED for v in df_stock['MACD_Hist']]
        fig_macd.add_trace(go.Bar(x=df_stock['Date'],y=df_stock['MACD_Hist'],marker_color=mc,name="Histogram"))
        fig_macd.add_trace(go.Scatter(x=df_stock['Date'],y=df_stock['MACD'],line=dict(color=BLUE,width=1.5),name="MACD"))
        fig_macd.add_trace(go.Scatter(x=df_stock['Date'],y=df_stock['MACD_Signal'],line=dict(color=GOLD,width=1.5),name="Signal"))
        fig_macd.update_layout(**CHART_BASE,height=240)
        fig_macd.update_xaxes(gridcolor=GRID)
        st.plotly_chart(fig_macd,use_container_width=True)

# ════════ TAB 2 — SENTIMENT ════════
with tab2:
    # Merge sentiment with stock for display
    df_sent_plot = df_sent.copy()
    col_a,col_b = st.columns([2,1])
    with col_a:
        st.markdown('<div class="section-title">Daily Sentiment Score</div>',unsafe_allow_html=True)
        fig_s = go.Figure()
        sc = [GREEN if v>0 else RED for v in df_sent_plot['Sentiment']]
        fig_s.add_trace(go.Bar(x=df_sent_plot['Date'],y=df_sent_plot['Sentiment'],marker_color=sc,name="Score"))
        ma7 = df_sent_plot['Sentiment'].rolling(7).mean()
        fig_s.add_trace(go.Scatter(x=df_sent_plot['Date'],y=ma7,line=dict(color=GOLD,width=2),name="7-day MA"))
        fig_s.add_hline(y=0,line_color='#8b949e',line_dash='dot')
        fig_s.update_layout(**CHART_BASE,height=300)
        fig_s.update_xaxes(gridcolor=GRID)
        st.plotly_chart(fig_s,use_container_width=True)
    with col_b:
        st.markdown('<div class="section-title">Overall Breakdown</div>',unsafe_allow_html=True)
        pos = (df_sent_plot['Sentiment']>0).mean()
        neg = (df_sent_plot['Sentiment']<0).mean()
        neu = (df_sent_plot['Sentiment']==0).mean()
        fig_pie = go.Figure(go.Pie(
            labels=['Positive','Negative','Neutral'],
            values=[pos,neg,neu],hole=0.55,
            marker_colors=[GREEN,RED,GOLD]))
        fig_pie.update_layout(paper_bgcolor=CARD,font_color='#c9d1d9',
            height=300,margin=dict(l=0,r=0,t=10,b=0),legend=dict(bgcolor=CARD))
        st.plotly_chart(fig_pie,use_container_width=True)

    st.markdown('<div class="section-title">Sentiment Statistics</div>',unsafe_allow_html=True)
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Average Sentiment", f"{df_sent_plot['Sentiment'].mean():+.3f}")
    s2.metric("Max Positive",      f"{df_sent_plot['Sentiment'].max():+.3f}")
    s3.metric("Max Negative",      f"{df_sent_plot['Sentiment'].min():+.3f}")
    s4.metric("Positive Days",     f"{(df_sent_plot['Sentiment']>0).sum()}")

# ════════ TAB 3 — PREDICTIONS ════════
with tab3:
    st.markdown('<div class="section-title">All Models — Predicted vs Actual (Test Period)</div>',unsafe_allow_html=True)

    model_choice = st.radio("Show model:", ["All Models","LSTM Only","Prophet Only","ARIMA Only"], horizontal=True)

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=df_pred['Date'],y=df_pred['Actual'],
        name="Actual Price",line=dict(color=BLUE,width=2.5)))

    if model_choice in ["All Models","LSTM Only"] and 'LSTM_Pred' in df_pred.columns:
        fig_p.add_trace(go.Scatter(x=df_pred['Date'],y=df_pred['LSTM_Pred'],
            name="LSTM",line=dict(color=GREEN,width=2,dash='dash')))
    if model_choice in ["All Models","Prophet Only"] and 'Prophet_Pred' in df_pred.columns:
        fig_p.add_trace(go.Scatter(x=df_pred['Date'],y=df_pred['Prophet_Pred'],
            name="Prophet",line=dict(color=GOLD,width=2,dash='dot')))
    if model_choice in ["All Models","ARIMA Only"] and 'ARIMA_Pred' in df_pred.columns:
        fig_p.add_trace(go.Scatter(x=df_pred['Date'],y=df_pred['ARIMA_Pred'],
            name="ARIMA",line=dict(color=PURP,width=2,dash='longdash')))

    fig_p.update_layout(**CHART_BASE,height=450)
    fig_p.update_xaxes(gridcolor=GRID)
    st.plotly_chart(fig_p,use_container_width=True)

    # Per-model error stats
    st.markdown('<div class="section-title">Model Error on Test Period</div>',unsafe_allow_html=True)
    cols = st.columns(3)
    model_cols = [('LSTM','LSTM_Pred',GREEN),('Prophet','Prophet_Pred',GOLD),('ARIMA','ARIMA_Pred',PURP)]
    for col,(mname,mcol,mcolor) in zip(cols,model_cols):
        if mcol in df_pred.columns:
            valid = df_pred.dropna(subset=[mcol,'Actual'])
            rmse  = np.sqrt(np.mean((valid['Actual']-valid[mcol])**2))
            mae   = np.mean(np.abs(valid['Actual']-valid[mcol]))
            col.markdown(f'<div class="info-box" style="border-color:{mcolor}"><b style="color:{mcolor}">{mname}</b><br>RMSE: ${rmse:.2f}<br>MAE: ${mae:.2f}</div>',unsafe_allow_html=True)

    # ── 🔮 FUTURE FORECAST (live, forward-looking) ─────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">🔮 Future Forecast — Next 30 Trading Days</div>',unsafe_allow_html=True)
    st.caption("✅ Loaded from future_forecast.csv (real model output)" if future_ok
               else "⚠️ future_forecast.csv not found — showing a live trend-based estimate from recent price action. Add the real file from Member 3 for the true model forecast.")

    tail_actual = df_stock.tail(30)

    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=tail_actual['Date'], y=tail_actual['Close'],
        name="Recent Actual", line=dict(color=BLUE, width=2.5)))

    if lstm_fcol:
        fig_f.add_trace(go.Scatter(x=df_future['Date'], y=df_future[lstm_fcol],
            name="LSTM Forecast", line=dict(color=GREEN, width=2, dash='dash')))
    if prophet_fcol:
        fig_f.add_trace(go.Scatter(x=df_future['Date'], y=df_future[prophet_fcol],
            name="Prophet Forecast", line=dict(color=GOLD, width=2, dash='dot')))
    if arima_fcol:
        fig_f.add_trace(go.Scatter(x=df_future['Date'], y=df_future[arima_fcol],
            name="ARIMA Forecast", line=dict(color=PURP, width=2, dash='longdash')))

    fig_f.add_vline(x=tail_actual['Date'].iloc[-1], line_color='#8b949e', line_dash='dot',
        annotation_text="Today", annotation_position="top")
    fig_f.update_layout(**CHART_BASE, height=380)
    fig_f.update_xaxes(gridcolor=GRID)
    st.plotly_chart(fig_f, use_container_width=True)

    fc1, fc2, fc3 = st.columns(3)
    if lstm_fcol:
        target = df_future[lstm_fcol].iloc[-1]
        fc1.markdown(f'<div class="info-box" style="border-color:{GREEN}"><b style="color:{GREEN}">LSTM — 30-Day Target</b><br>${target:.2f} ({(target/cur-1)*100:+.1f}%)</div>', unsafe_allow_html=True)
    if prophet_fcol:
        target = df_future[prophet_fcol].iloc[-1]
        fc2.markdown(f'<div class="info-box" style="border-color:{GOLD}"><b style="color:{GOLD}">Prophet — 30-Day Target</b><br>${target:.2f} ({(target/cur-1)*100:+.1f}%)</div>', unsafe_allow_html=True)
    if arima_fcol:
        target = df_future[arima_fcol].iloc[-1]
        fc3.markdown(f'<div class="info-box" style="border-color:{PURP}"><b style="color:{PURP}">ARIMA — 30-Day Target</b><br>${target:.2f} ({(target/cur-1)*100:+.1f}%)</div>', unsafe_allow_html=True)

# ════════ TAB 4 — BACKTESTING ════════
with tab4:
    df_bt = df_stock.copy()
    if 'MA20' in df_bt.columns and 'MA50' in df_bt.columns:
        d = df_bt.dropna(subset=['MA20','MA50']).copy()
        d['Signal']    = np.where(d['MA20']>d['MA50'],1,-1)
    elif 'MA10' in df_bt.columns and 'MA20' in df_bt.columns:
        d = df_bt.dropna(subset=['MA10','MA20']).copy()
        d['Signal']    = np.where(d['MA10']>d['MA20'],1,-1)
    else:
        d = df_bt.copy(); d['Signal'] = 1

    if len(d) > 2:
        d['Ret']       = d['Close'].pct_change()
        d['Strat_Ret'] = d['Signal'].shift(1)*d['Ret']
        d['Strat_Cum'] = (1+d['Strat_Ret'].fillna(0)).cumprod()
        d['BH_Cum']    = (1+d['Ret'].fillna(0)).cumprod()
        wins     = (d['Strat_Ret']>0).sum()
        total    = (d['Strat_Ret']!=0).sum()
        win_rate = wins/total*100 if total>0 else 0
        sharpe   = (d['Strat_Ret'].mean()/d['Strat_Ret'].std())*(252**0.5) if d['Strat_Ret'].std()>0 else 0
        fs = d['Strat_Cum'].iloc[-1]; fb = d['BH_Cum'].iloc[-1]
        dd = ((d['Strat_Cum']-d['Strat_Cum'].cummax())/d['Strat_Cum'].cummax()).min()

        b1,b2,b3,b4 = st.columns(4)
        b1.metric("Strategy Return",   f"{(fs-1)*100:.1f}%")
        b2.metric("Buy & Hold Return", f"{(fb-1)*100:.1f}%")
        b3.metric("Win Rate",          f"{win_rate:.1f}%")
        b4.metric("Sharpe Ratio",      f"{sharpe:.2f}")

        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(x=d['Date'],y=d['Strat_Cum'],
            name="MA Crossover",line=dict(color=GREEN,width=2.5),
            fill='tozeroy',fillcolor='rgba(63,185,80,0.07)'))
        fig_b.add_trace(go.Scatter(x=d['Date'],y=d['BH_Cum'],
            name="Buy & Hold",line=dict(color=BLUE,width=2,dash='dash')))
        fig_b.add_hline(y=1,line_color='#8b949e',line_dash='dot',annotation_text="Break-even")
        fig_b.update_layout(**CHART_BASE,height=360)
        fig_b.update_xaxes(gridcolor=GRID)
        st.plotly_chart(fig_b,use_container_width=True)

        drawdown = (d['Strat_Cum']-d['Strat_Cum'].cummax())/d['Strat_Cum'].cummax()
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=d['Date'],y=drawdown*100,
            fill='tozeroy',fillcolor='rgba(248,81,73,0.2)',
            line=dict(color=RED,width=1.5),name="Drawdown (%)"))
        fig_dd.update_layout(**CHART_BASE,height=220)
        fig_dd.update_xaxes(gridcolor=GRID)
        st.plotly_chart(fig_dd,use_container_width=True)
        st.markdown(f'<div class="info-box">📌 Max Drawdown: {dd*100:.1f}%</div>',unsafe_allow_html=True)

# ════════ TAB 5 — MODEL EVALUATION ════════
with tab5:
    st.markdown('<div class="section-title">Model Performance Comparison</div>',unsafe_allow_html=True)

    # Metrics table
    st.dataframe(df_eval.style.highlight_max(subset=['R2 Score'],color='#0d3321')
                              .highlight_min(subset=['RMSE','MAE'],color='#0d3321'),
                 use_container_width=True, hide_index=True)

    col_x,col_y = st.columns(2)
    with col_x:
        st.markdown('<div class="section-title">RMSE Comparison (lower = better)</div>',unsafe_allow_html=True)
        colors_bar = [GREEN if m=='LSTM' else GOLD if m=='Prophet' else PURP for m in df_eval['Model']]
        fig_rmse = go.Figure(go.Bar(x=df_eval['Model'],y=df_eval['RMSE'],
            marker_color=colors_bar,text=df_eval['RMSE'].round(2),textposition='outside'))
        fig_rmse.update_layout(**CHART_BASE,height=300,showlegend=False)
        fig_rmse.update_xaxes(gridcolor=GRID)
        st.plotly_chart(fig_rmse,use_container_width=True)

    with col_y:
        st.markdown('<div class="section-title">R² Score Comparison (higher = better)</div>',unsafe_allow_html=True)
        fig_r2 = go.Figure(go.Bar(x=df_eval['Model'],y=df_eval['R2 Score'],
            marker_color=colors_bar,text=df_eval['R2 Score'].round(3),textposition='outside'))
        fig_r2.add_hline(y=0,line_color='#8b949e',line_dash='dot')
        fig_r2.update_layout(**CHART_BASE,height=300,showlegend=False)
        fig_r2.update_xaxes(gridcolor=GRID)
        st.plotly_chart(fig_r2,use_container_width=True)

    st.markdown('<div class="section-title">Winner</div>',unsafe_allow_html=True)
    st.markdown(f'''<div class="info-box">
    🏆 <b style="color:{GREEN}">LSTM wins</b> with R² = 0.925 (92.5% accuracy) and RMSE of just $5.08<br><br>
    📉 Prophet had moderate performance — RMSE $31.39, R² -1.58 (below baseline)<br>
    ❌ ARIMA performed poorly — RMSE $45.33, R² -4.38 (much worse than baseline)<br><br>
    💡 LSTM is the recommended model for this stock prediction system.
    </div>''',unsafe_allow_html=True)

    st.markdown("### 👥 Team")
    c1,c2,c3,c4 = st.columns(4)
    for col,name,role,tech in [
        (c1,"Member 1","Data Collection",   "yfinance, Pandas, ta"),
        (c2,"Member 2","Sentiment Analysis","NewsAPI, VADER"),
        (c3,"Member 3","Prediction Models", "LSTM, Prophet, ARIMA"),
        (c4,"Member 4","Frontend & Backtest","Streamlit, Plotly"),
    ]:
        col.markdown(f'<div class="info-box"><b>{name}</b><br><span style="color:#8b949e;font-size:13px">{role}</span><br><br><span style="background:#21262d;color:#8b949e;border-radius:6px;padding:2px 8px;font-size:12px">{tech}</span></div>',unsafe_allow_html=True)