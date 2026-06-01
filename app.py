import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
import streamlit as st
import yfinance as yf

# Set up the web page configurations
st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")
st.title("📈 Stock Price Forecasting & Fundamental Analysis Dashboard")

# Sidebar Configurations
st.sidebar.header("Dashboard Controls")
forecast_days = st.sidebar.slider(
    "Forecast Horizon (Days)", min_value=30, max_value=180, value=90, step=15
)
rsi_window = st.sidebar.slider(
    "RSI Calculation Window (Days)", min_value=7, max_value=21, value=14
)

# Ticker List
tickers = {
    "Google": "GOOG",
    "Palantir": "PLTR",
    "Microsoft": "MSFT",
    "Intel": "INTC",
    "Tesla": "TSLA",
    "Nokia": "NOK",
    "Nvidia": "NVDA",
    "AMD": "AMD",
    "Apple": "AAPL",
    "Taiwan Semi": "TSM",
}


# --- CACHED FUNCTIONS TO PREVENT YFINANCE RATE LIMITS ---
# ttl=3600 caches the market data for exactly 1 hour (3600 seconds)
@st.cache_data(ttl=3600)
def fetch_stock_history(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.reset_index()


@st.cache_data(ttl=3600)
def fetch_stock_fundamentals(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        return ticker_obj.info
    except Exception:
        return {}


# Helper function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


# Add manual button which clears the cache when clicked
if st.sidebar.button("🔄 Force Clear Cache & Refresh Data"):
    st.cache_data.clear()

start_date = "2016-01-01"
end_date = datetime.date.today().strftime("%Y-%m-%d")

summary_data = []
charts_dict = {}

with st.spinner("Analyzing live market trends safely via data cache..."):
    for name, ticker in tickers.items():
        # 1. Fetch historical data safely via cache
        df = fetch_stock_history(ticker, start=start_date, end=end_date)
        if df is None or df.empty:
            continue

        # 2. Fetch fundamentals safely via cache
        info = fetch_stock_fundamentals(ticker)

        # Extract fundamental metrics
        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        peg_ratio = info.get("pegRatio")
        ps_ratio = info.get("priceToSalesTrailing12Months")
        roe = info.get("returnOnEquity")
        op_margin = info.get("operatingMargins")

        # Fetch and convert Debt-to-Equity
        raw_de = info.get("debtToEquity")
        de_ratio = round(raw_de / 100, 2) if raw_de is not None else "N/A"

        # Convert fractional percentages
        roe_formatted = f"{roe * 100:.2f}%" if roe is not None else "N/A"
        margin_formatted = (
            f"{op_margin * 100:.2f}%" if op_margin is not None else "N/A"
        )

        # Technical Analysis
        df["RSI"] = calculate_rsi(df["Close"], window=rsi_window)
        latest_rsi = df["RSI"].iloc[-1]
        latest_close = df["Close"].iloc[-1]

        # Prophet Modeling
        prophet_df = df[["Date", "Close"]].rename(
            columns={"Date": "ds", "Close": "y"}
        )
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)

        model = Prophet(daily_seasonality=False, weekly_seasonality=True)
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        future_predicted = forecast.iloc[-1]
        pred_change_pct = (
            (future_predicted["yhat"] - latest_close) / latest_close
        ) * 100

        # Signal Matrix
        if latest_rsi < 35 and pred_change_pct > 0 and pe_ratio is not None and pe_ratio < 80:
            signal = "🟢 Strong Buy"
        elif (latest_rsi < 45 and pe_ratio is not None and pe_ratio < 80) or (
            pred_change_pct > 5 and pe_ratio is not None and pe_ratio < 80):
            signal = "🟡 Accumulate"
        elif latest_rsi > 70:
            signal = "🔴 Overbought"
        else:
            signal = "⚪ Hold"

        summary_data.append(
            {
                "Company": name,
                "Ticker": ticker,
                "Price ($)": round(latest_close, 2),
				f"Forecasted Price ({forecast_days}d)": round(
                        future_predicted["yhat"], 2),
                "RSI": round(latest_rsi, 2),
                "P/E": round(pe_ratio, 2) if pe_ratio else "N/A",
                "PEG": round(peg_ratio, 2) if peg_ratio else "N/A",
                "P/S": round(ps_ratio, 2) if ps_ratio else "N/A",
                "ROE": roe_formatted,
                "Op. Margin": margin_formatted,
                "D/E Ratio": de_ratio,
                "Expected Move (%)": round(pred_change_pct, 2),
                "Action Signal": signal,
            }
        )

        # Build Figure Plots
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            prophet_df["ds"],
            prophet_df["y"],
            label="Historical Close",
            color="#2F3E46",
        )
        ax.plot(
            forecast["ds"],
            forecast["yhat"],
            label="Prophet Forecast",
            color="#0077B6",
            linestyle="--",
        )
        ax.fill_between(
            forecast["ds"],
            forecast["yhat_lower"],
            forecast["yhat_upper"],
            color="#0077B6",
            alpha=0.15,
        )
        ax.set_title(f"{name} ({ticker}) Forecast Trend")
        ax.grid(True, alpha=0.2)
        ax.legend()
        charts_dict[name] = fig

summary_df = pd.DataFrame(summary_data)

# ---- RENDER WEB GRAPHICS INTERFACE ----
st.subheader("📊 Global Valuation & Investment Strategy Matrix")
st.dataframe(summary_df, use_container_width=True)

st.subheader("📈 Individual Stock Forecasting Breakdowns")
tabs = st.tabs(list(tickers.keys()))
for index, name in enumerate(tickers.keys()):
    with tabs[index]:
        st.write(f"### {name} Predictions")
        if name in charts_dict:
            st.pyplot(charts_dict[name])
