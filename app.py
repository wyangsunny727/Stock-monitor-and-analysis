import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
import streamlit as st
import yfinance as yf

# Set up the web page configurations
st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")
st.title("📈 Stock Price Forecasting & Investment Strategy Dashboard")
st.write(
    "This dashboard pulls real-time data from Yahoo Finance and utilizes Meta's Prophet model to evaluate optimal buy windows."
)

# Sidebar Configurations
st.sidebar.header("Dashboard Controls")
forecast_days = st.sidebar.slider(
    "Forecast Horizon (Days)", min_value=30, max_value=180, value=90, step=15
)
rsi_window = st.sidebar.slider(
    "RSI Calculation Window (Days)", min_value=7, max_value=21, value=14
)

# Predefined Tickers
tickers = {
    "Google": "GOOG",
    "Palantir": "PLTR",
    "Microsoft": "MSFT",
    "Intel": "INTC",
    "Tesla": "TSLA",
    "Nokia": "NOK",
    "Nvidia": "NVDA",
    "AMD": "AMD",
    "Micron Technology": "MU",
    "Taiwan Semi": "TSM",
    
}


# Helper function to calculate RSI
def calculate_rsi(data, window=21):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Add a manual refresh trigger button in the sidebar
refresh_button = st.sidebar.button("🔄 Refresh Data & Re-run Forecast")

# Streamlit naturally re-runs the entire script when the page loads,
# or when any input widget changes / button is clicked.
if refresh_button or "initialized" not in st.session_state:
    st.session_state["initialized"] = True

    start_date = "2016-01-01"
    end_date = datetime.date.today().strftime("%Y-%m-%d")

    summary_data = []
    charts_dict = {}

    # Status indicator loading spinner
    with st.spinner("Fetching live market data and calculating forecasts..."):
        for name, ticker in tickers.items():
            df = yf.download(ticker, start=start_date, end=end_date)
            if df.empty:
                continue

            # Handle multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()

            # Technical Analysis Metrics
            df["RSI"] = calculate_rsi(df["Close"], window=rsi_window)
            latest_rsi = df["RSI"].iloc[-1]
            latest_close = df["Close"].iloc[-1]

            # Machine Learning Modeling (Prophet)
            prophet_df = df[["Date", "Close"]].rename(
                columns={"Date": "ds", "Close": "y"}
            )
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(
                None
            )

            model = Prophet(daily_seasonality=False, weekly_seasonality=True)
            model.fit(prophet_df)

            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)

            future_predicted = forecast.iloc[-1]
            pred_change_pct = (
                (future_predicted["yhat"] - latest_close) / latest_close
            ) * 100

            # Dynamic Decision Rules Matrix
            if latest_rsi < 35 and pred_change_pct > 0:
                signal = "🟢 Strong Buy (Oversold & Upward Trend)"
            elif latest_rsi < 45 or pred_change_pct > 5:
                signal = "🟡 Accumulate / Buy Dip"
            elif latest_rsi > 70:
                signal = "🔴 Overbought (Wait for Pullback)"
            else:
                signal = "⚪ Hold / Neutral"

            summary_data.append(
                {
                    "Company": name,
                    "Ticker": ticker,
                    "Current Price ($)": round(latest_close, 2),
                    f"RSI ({rsi_window}d)": round(latest_rsi, 2),
                    f"Forecasted Price ({forecast_days}d)": round(
                        future_predicted["yhat"], 2
                    ),
                    "Expected Move (%)": round(pred_change_pct, 2),
                    "Action Signal": signal,
                }
            )

            # Build Figure Plotly or Matplotlib
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

    # Save outputs to session state so they persist smoothly
    st.session_state["summary_df"] = pd.DataFrame(summary_data)
    st.session_state["charts"] = charts_dict

# ---- RENDER WEB GRAPHICS INTERFACE ----
if "summary_df" in st.session_state:
    st.subheader("📊 US Tech Stock Matrix")
    # Display an interactive data table with highlighted rows
    st.dataframe(st.session_state["summary_df"], use_container_width=True)

    st.subheader("📈 Individual Stock Forecasting Breakdowns")
    # Create tabs for clean browsing experience between stocks
    tabs = st.tabs(list(tickers.keys()))
    for index, name in enumerate(tickers.keys()):
        with tabs[index]:
            st.write(f"### {name} Predictions")
            if name in st.session_state["charts"]:
                st.pyplot(st.session_state["charts"][name])