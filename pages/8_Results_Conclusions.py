import streamlit as st
import pandas as pd
import os

EVAL_PATH = "models/evaluation_results.csv"


if not os.path.exists(EVAL_PATH):
    st.error("❌ evaluation_results.csv not found")
    st.stop()

df = pd.read_csv(EVAL_PATH)


st.title("Results & Conclusions")

st.markdown(
    """
    This section summarises the **quantitative evaluation results**, explicitly
    defines the **Key Performance Indicators (KPIs)** used, and provides an
    **interpretation of model performance** across cryptocurrencies.
    """
)


st.subheader("Key Performance Indicators (KPIs)")

st.markdown(
    """
    To evaluate and compare the forecasting models, a set of **Key Performance Indicators (KPIs)**
    was defined to capture prediction accuracy, robustness, and explanatory power.

    **KPI 1 – Mean Absolute Error (MAE)**  
    MAE measures the average absolute difference between predicted and actual prices.
    It provides a clear and interpretable indication of typical forecast error in price units.
    Lower MAE values indicate higher predictive accuracy.

    **KPI 2 – Root Mean Squared Error (RMSE)**  
    RMSE penalises larger prediction errors more heavily than MAE and is particularly
    important in volatile markets such as cryptocurrencies. This KPI reflects model stability
    and sensitivity to extreme price movements.

    **KPI 3 – Mean Absolute Percentage Error (MAPE)**  
    MAPE expresses forecasting error as a percentage of the actual price, enabling
    scale-independent comparison across cryptocurrencies with vastly different price levels.

    **KPI 4 – Coefficient of Determination (R² Score)**  
    The R² score measures the proportion of variance in actual prices explained by the model.
    It reflects how well the model captures underlying price dynamics rather than just point accuracy.

    **KPI 5 – Model Confidence Level (Derived KPI)**  
    Model confidence is derived from historical forecasting accuracy and is calculated as:
    **Model Confidence = 100 − MAPE**.  
    This KPI provides an intuitive summary of historical model reliability.
    """
)


st.subheader("Best Performing Model per Cryptocurrency (Based on RMSE)")

best_models = (
    df.sort_values("RMSE")
      .groupby("Symbol")
      .first()
      .reset_index()
)

st.dataframe(
    best_models[["Symbol", "Model", "RMSE", "MAE", "MAPE (%)", "R2"]],
    use_container_width=True
)

st.subheader("Model Performance Interpretation")

st.markdown(
    """
    **Key observations from KPI analysis:**

    - **Random Forest** achieves high historical accuracy on several assets due to its
      ability to model non-linear relationships, though its long-horizon forecasts are limited.
    - **LSTM** demonstrates strong performance on assets with pronounced temporal dependencies,
      benefiting from its sequential learning capability.
    - **ARIMA** provides stable and interpretable forecasts but struggles during abrupt market shifts.
    - **Prophet** produces smooth forecasts that prioritise trend estimation over short-term volatility.

    These results indicate that **model effectiveness varies by asset and forecasting horizon**.
    """
)


st.subheader("Limitations")

st.markdown(
    """
    - Forecast accuracy decreases as the prediction horizon increases.
    - Tree-based models such as Random Forest have limited extrapolation capability.
    - Prophet assumes additive trend structures, reducing responsiveness to sharp spikes.
    - The analysis does not incorporate exogenous variables such as macroeconomic or sentiment data.
    """
)


st.subheader("Future Work")

st.markdown(
    """
    - Incorporate **exogenous variables** such as sentiment indicators or macroeconomic signals.
    - Explore **ensemble approaches** combining statistical and deep learning models.
    - Introduce **probabilistic forecasting** and confidence intervals.
    - Apply **rolling-window validation** to further assess model robustness.
    """
)


st.subheader("Final Conclusion")

st.markdown(
    """
    This study demonstrates that no single forecasting model consistently outperforms others
    across all cryptocurrencies. Performance varies depending on asset characteristics and
    forecasting horizon. The explicit use of KPIs enables objective comparison and informed
    interpretation, reinforcing the importance of **data-driven model selection** in
    cryptocurrency price forecasting.
    """
)
