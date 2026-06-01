import datetime
import time  # For safe API throttling
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
    "Forecast Horizon (Days)", min_value=30, max_value=360, value=90, step=15
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
    "Amazon": "AMZN",
    "Taiwan Semi": "TSM",
}


# --- CACHED FUNCTIONS TO PREVENT YFINANCE RATE LIMITS ---
@st.cache_data(ttl=3600)
def fetch_all_historical_data(ticker_dict, start, end):
    """Downloads historical data for ALL tickers simultaneously in 1 network request"""
    symbols = list(ticker_dict.values())
    all_data = yf.download(symbols, start=start, end=end)
    return all_data


@st.cache_data(ttl=3600)
def fetch_stock_fundamentals_and_calendar(ticker):
    """Fetches fundamental info safely using custom request sessions to bypass blocks"""
    try:
        time.sleep(0.5)  # Polite pause to prevent rate limits
        
        # FIX: Create a session with a realistic User-Agent to prevent getting an empty dict
        import requests
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        ticker_obj = yf.Ticker(ticker, session=session)
        info_dict = ticker_obj.info if ticker_obj.info else {}

        # Safely attempt parsing next earnings date
        earnings_date_str = "N/A"
        try:
            if hasattr(ticker_obj, "calendar") and ticker_obj.calendar is not None:
                cal = ticker_obj.calendar
                if isinstance(cal, dict) and "Earnings Date" in cal:
                    dates = cal["Earnings Date"]
                    if dates:
                        earnings_date_str = dates[0].strftime("%Y-%m-%d")
                elif isinstance(cal, pd.DataFrame) and "Value" in cal.index:
                    earnings_date_str = cal.loc["Earnings Date"].values[0].strftime("%Y-%m-%d")
        except Exception:
            if "calendarOutputs" in info_dict:
                cal_out = info_dict.get("calendarOutputs", {})
                if "earningsDate" in cal_out and cal_out["earningsDate"]:
                    earnings_date_str = cal_out["earningsDate"][0]

        info_dict["_next_earnings"] = earnings_date_str
        return info_dict
    except Exception:
        return {"_next_earnings": "N/A"}


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
    # Step 1: Bulk historical download
    bulk_historical = fetch_all_historical_data(tickers, start=start_date, end=end_date)

    # Step 2: Process metrics
    for name, ticker in tickers.items():
        # Extract individual slice from bulk download
        if isinstance(bulk_historical.columns, pd.MultiIndex):
            try:
                df = bulk_historical.xs(ticker, level=1, axis=1).reset_index()
            except KeyError:
                continue
        else:
            df = bulk_historical.reset_index()

        if df.empty or "Close" not in df.columns:
            continue

        df = df.dropna(subset=["Close"])
        latest_close = float(df["Close"].iloc[-1])

        # Fetch fundamentals and calendar timelines securely
        info = fetch_stock_fundamentals_and_calendar(ticker)
        next_earnings = info.get("_next_earnings", "N/A")

        # Extract fundamental metrics with manual calculation backup
        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        
        # BACKUP: Calculate P/E using price and EPS if trailingPE is missing
        if pe_ratio is None or pe_ratio == "N/A":
            eps = info.get("trailingEps") or info.get("forwardEps")
            if eps and eps > 0:
                pe_ratio = latest_close / eps

        peg_ratio = info.get("pegRatio")
        ps_ratio = info.get("priceToSalesTrailing12Months")
        roe = info.get("returnOnEquity")
        op_margin = info.get("operatingMargins")

        # Fetch and convert Debt-to-Equity
        raw_de = info.get("debtToEquity")
        de_ratio = round(raw_de / 100, 2) if raw_de is not None else "N/A"

        # Convert fractional percentages
        roe_formatted = f"{roe * 100:.2f}%" if roe is not None else "N/A"
        margin_formatted = f"{op_margin * 100:.2f}%" if op_margin is not None else "N/A"

        # Technical Analysis
        df["RSI"] = calculate_rsi(df["Close"], window=rsi_window)
        latest_rsi = df["RSI"].iloc[-1]

        # Prophet Modeling
        prophet_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)

        model = Prophet(daily_seasonality=False, weekly_seasonality=True)
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        future_predicted = forecast.iloc[-1]
        pred_change_pct = ((future_predicted["yhat"] - latest_close) / latest_close) * 100

        # Safe logical evaluations
        has_safe_pe = pe_ratio is not None and pe_ratio < 80

        if latest_rsi < 35 and pred_change_pct > 0 and has_safe_pe:
            signal = "🟢 Strong Buy"
        elif (latest_rsi < 45 and has_safe_pe) or (pred_change_pct > 5 and has_safe_pe):
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
                f"Forecasted Price ({forecast_days}d)": round(future_predicted["yhat"], 2),
                "Next Earnings": next_earnings,
                "RSI": round(latest_rsi, 2),
                "P/E": round(pe_ratio, 2) if isinstance(pe_ratio, (int, float)) else "N/A",
                "PEG (5Y Expected)": round(peg_ratio, 2) if peg_ratio else "N/A",
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
        ax.plot(prophet_df["ds"], prophet_df["y"], label="Historical Close", color="#2F3E46")
        ax.plot(forecast["ds"], forecast["yhat"], label="Prophet Forecast", color="#0077B6", linestyle="--")
        ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="#0077B6", alpha=0.15)
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
