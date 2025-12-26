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
    "14 Days": 14,
    "1 Month (30 Days)": 30,
    "6 Months (180 Days)": 180
}


st.title("💰 Profit / Investment Planner")

st.markdown(
    """
    This page helps users **identify optimal buy and sell points** using forecasted prices
    and estimate **potential profit or loss** under different investment scenarios.
    """
)


coin = st.selectbox("Select Cryptocurrency", coins)
model_name = st.selectbox("Select Forecasting Model", list(models.keys()))
horizon_label = st.selectbox("Select Planning Horizon", list(horizon_map.keys()))

investment_amount = st.number_input(
    "Investment Amount (USD)",
    min_value=100.0,
    value=1000.0,
    step=100.0
)

horizon_days = horizon_map[horizon_label]
forecast_file = f"{coin}_{models[model_name]}.csv"
forecast_path = os.path.join(MODELS_PATH, forecast_file)


if not os.path.exists(forecast_path):
    st.error("❌ Forecast data not found for the selected options.")
    st.stop()

forecast_df = pd.read_csv(forecast_path, parse_dates=["Date"])

forecast_df = forecast_df[forecast_df["Day_Number"] <= horizon_days]


buy_row = forecast_df.loc[forecast_df["Forecast_Close"].idxmin()]
sell_row = forecast_df.loc[forecast_df["Forecast_Close"].idxmax()]

buy_price = buy_row["Forecast_Close"]
sell_price = sell_row["Forecast_Close"]

buy_date = buy_row["Date"]
sell_date = sell_row["Date"]


units_bought = investment_amount / buy_price
final_value = units_bought * sell_price

profit_loss = final_value - investment_amount
profit_loss_pct = (profit_loss / investment_amount) * 100


st.subheader("📅 Optimal Buy & Sell Dates (Forecast-Based)")

col1, col2 = st.columns(2)

col1.metric(
    "Best Buy Date",
    buy_date.strftime("%Y-%m-%d"),
    f"${buy_price:.2f}"
)

col2.metric(
    "Best Sell Date",
    sell_date.strftime("%Y-%m-%d"),
    f"${sell_price:.2f}"
)

st.subheader("📈 Expected Investment Outcome")

col3, col4 = st.columns(2)

col3.metric(
    "Expected Profit / Loss",
    f"${profit_loss:,.2f}",
    f"{profit_loss_pct:.2f} %"
)

col4.metric(
    "Estimated Final Value",
    f"${final_value:,.2f}"
)


st.subheader("🔍 What-If Scenario Explanation")

st.markdown(
    f"""
    - The model assumes an investment of **${investment_amount:,.2f}**
      made at the **lowest forecasted price** within the selected horizon.
    - The asset is sold at the **highest forecasted price** within the same period.
    - No transaction costs or slippage are considered.
    - This scenario is **purely forecast-based** and does not represent financial advice.
    """
)


st.subheader("📄 Forecast Prices Used")

st.dataframe(
    forecast_df[["Date", "Forecast_Close"]],
    use_container_width=True
)
