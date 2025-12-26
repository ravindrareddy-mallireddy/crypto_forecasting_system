import streamlit as st
import pandas as pd
import os


MODELS_PATH = "models"

coins = ["BTC-USD", "ETH-USD", "XRP-USD", "AVAX-USD"]

models = {
    "Random Forest": "rf_forecast_next_6_months",
    "ARIMA": "arima_forecast_next_6_months",
    "LSTM": "lstm_forecast_next_6_months",
    "Prophet": "prophet_forecast_next_6_months"
}

horizon_map = {
    "7 Days": 7,
    "14 Days": 14
}


st.title("🌍 Market Overview")

st.markdown(
    """
    This page provides a **group-level market direction analysis** by aggregating
    short-term cryptocurrency forecasts.  
    The objective is to identify whether the **overall market trend** is expected
    to move **upward or downward** based on majority forecast signals.
    """
)


model_name = st.selectbox("Select Forecasting Model", list(models.keys()))
horizon_label = st.selectbox("Select Short-Term Horizon", list(horizon_map.keys()))

horizon_days = horizon_map[horizon_label]


rows = []
up_count = 0
down_count = 0

for coin in coins:
    forecast_file = f"{coin}_{models[model_name]}.csv"
    forecast_path = os.path.join(MODELS_PATH, forecast_file)

    if not os.path.exists(forecast_path):
        continue

    forecast_df = pd.read_csv(forecast_path, parse_dates=["Date"])
    forecast_df = forecast_df[forecast_df["Day_Number"] == horizon_days]

    if forecast_df.empty:
        continue

    forecast_price = forecast_df["Forecast_Close"].values[0]

    past_file = f"{coin}_{models[model_name].replace('forecast_next_6_months', 'past_predictions')}.csv"
    past_path = os.path.join(MODELS_PATH, past_file)

    if not os.path.exists(past_path):
        continue

    past_df = pd.read_csv(past_path)
    last_actual_price = past_df["Close"].iloc[-1]

    pct_change = ((forecast_price - last_actual_price) / last_actual_price) * 100

    if pct_change > 0:
        direction = "⬆️ Up"
        up_count += 1
    else:
        direction = "⬇️ Down"
        down_count += 1

    rows.append({
        "Coin": coin,
        "Last Actual Price": round(last_actual_price, 2),
        f"Forecast Price ({horizon_label})": round(forecast_price, 2),
        "Expected Change (%)": round(pct_change, 2),
        "Direction": direction
    })

market_df = pd.DataFrame(rows)


if up_count > down_count:
    market_trend = "📈 MARKET TREND: UP"
    trend_color = "green"
elif down_count > up_count:
    market_trend = "📉 MARKET TREND: DOWN"
    trend_color = "red"
else:
    market_trend = "➖ MARKET TREND: NEUTRAL"
    trend_color = "gray"


st.subheader("📊 Aggregate Market Direction")

st.markdown(
    f"""
    <h2 style="color:{trend_color}; text-align:center;">
        {market_trend}
    </h2>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Market direction is determined using a **majority-vote aggregation** of short-term forecasts."
)


st.subheader("🧾 Coin-Level Contribution")

st.dataframe(
    market_df,
    use_container_width=True
)


st.subheader("🧠 Interpretation")

st.markdown(
    f"""
    - The market trend is derived from **{horizon_label.lower()} forecasts**.
    - If the **majority of selected cryptocurrencies** show positive expected change,
      the overall market direction is classified as **UP**.
    - This analysis provides **group-level decision support** rather than individual
      trading signals.
    """
)


st.caption(
    "This market overview is a **forecast-based analytical tool** and does not constitute financial advice."
)
