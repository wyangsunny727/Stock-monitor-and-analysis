import datetime
import time  # Imported for throttling
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
import streamlit as st
import yfinance as yf

# ... Keep your page config, sidebar sliders, and calculation functions the same ...

# Expand your list as much as you want now!
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
    "Netflix": "NFLX",
}

start_date = "2016-01-01"
end_date = datetime.date.today().strftime("%Y-%m-%d")


# --- NEW GENERATION BATCH FETCHING TO BYPASS RATE LIMITS ---
@st.cache_data(ttl=3600)
def fetch_all_historical_data(ticker_dict, start, end):
    """Downloads historical data for ALL tickers simultaneously in 1 network request"""
    symbols = list(ticker_dict.values())
    # This downloads everything at once, drastically reducing network hits
    all_data = yf.download(symbols, start=start, end=end)
    return all_data


@st.cache_data(ttl=3600)
def fetch_single_fundamental(ticker):
    """Fetches info for one ticker, safely falling back if blocked"""
    try:
        # Add a tiny 0.5-second break so we don't bombard Yahoo's servers
        time.sleep(0.5)
        return yf.Ticker(ticker).info
    except Exception:
        return {}


# Add manual button which clears the cache when clicked
if st.sidebar.button("🔄 Force Clear Cache & Refresh Data"):
    st.cache_data.clear()

summary_data = []
charts_dict = {}

with st.spinner("Processing batch financial matrix safely..."):
    # Step 1: Download ALL historical data in ONE shot
    bulk_historical = fetch_all_historical_data(
        tickers, start=start_date, end=end_date
    )

    # Step 2: Loop through to process calculations
    for name, ticker in tickers.items():

        # Extract this stock's specific historical data from the bulk download
        if isinstance(bulk_historical.columns, pd.MultiIndex):
            # If multi-index, extract just the Close column for this ticker
            try:
                df_stock = bulk_historical.xs(ticker, level=1, axis=1).reset_index()
            except KeyError:
                continue
        else:
            # Fallback if only one stock was downloaded
            df_stock = bulk_historical.reset_index()

        if df_stock.empty or "Close" not in df_stock.columns:
            continue

        # Drop any rows where Close is NaN (e.g., if a stock didn't exist in 2018)
        df_stock = df_stock.dropna(subset=["Close"])

        # Fetch fundamentals with built-in sleep throttling
        info = fetch_single_fundamental(ticker)

        # Extract fundamental metrics
        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        peg_ratio = info.get("pegRatio")
        ps_ratio = info.get("priceToSalesTrailing12Months")
        roe = info.get("returnOnEquity")
        op_margin = info.get("operatingMargins")

        raw_de = info.get("debtToEquity")
        de_ratio = round(raw_de / 100, 2) if raw_de is not None else "N/A"

        roe_formatted = f"{roe * 100:.2f}%" if roe is not None else "N/A"
        margin_formatted = (
            f"{op_margin * 100:.2f}%" if op_margin is not None else "N/A"
        )

        # Technical Analysis (RSI) using the local extracted dataframe
        df_stock["RSI"] = calculate_rsi(df_stock["Close"], window=rsi_window)
        latest_rsi = df_stock["RSI"].iloc[-1]
        latest_close = df_stock["Close"].iloc[-1]

        # Prophet Modeling
        prophet_df = df_stock[["Date", "Close"]].rename(
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

        # Safe Signal Matrix Evaluation
        has_good_pe = pe_ratio is not None and pe_ratio < 80
        if latest_rsi < 35 and pred_change_pct > 0 and has_good_pe:
            signal = "🟢 Strong Buy"
        elif (latest_rsi < 45 and has_good_pe) or (
            pred_change_pct > 5 and has_good_pe
        ):
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
