import streamlit as st
import pandas as pd
import os

# ============================================================
# CONFIG
# ============================================================

EVAL_PATH = "models/evaluation_results.csv"

# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(EVAL_PATH):
    st.error("❌ evaluation_results.csv not found")
    st.stop()

df = pd.read_csv(EVAL_PATH)

# ============================================================
# PAGE TITLE
# ============================================================

st.title("📌 Results & Conclusions")

st.markdown(
    """
    This section summarises the **quantitative evaluation results** and provides
    **interpretation of model performance** across cryptocurrencies.
    """
)

# ============================================================
# BEST MODEL PER COIN (BASED ON RMSE)
# ============================================================

st.subheader("🏆 Best Performing Model per Cryptocurrency (RMSE)")

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

# ============================================================
# INTERPRETATION
# ============================================================

st.subheader("📊 Model Performance Interpretation")

st.markdown(
    """
    **Key observations:**

    - **LSTM** generally performs better on assets with stronger temporal dependency
      due to its ability to capture sequential patterns.
    - **ARIMA** provides stable forecasts but struggles with sudden regime changes.
    - **Random Forest** performs reasonably on lower-priced assets but struggles with
      large-scale trending series such as BTC.
    - **Prophet** produces smooth, conservative forecasts, prioritising trend stability
      over short-term volatility.
    """
)

# ============================================================
# LIMITATIONS
# ============================================================

st.subheader("⚠️ Limitations")

st.markdown(
    """
    - Forecasting accuracy decreases significantly for longer horizons.
    - Tree-based models lack extrapolation capability for strong trends.
    - Prophet assumes additive trend structures, which limits responsiveness to spikes.
    - No exogenous variables (macro, sentiment) were included.
    """
)

# ============================================================
# FUTURE WORK
# ============================================================

st.subheader("🚀 Future Work")

st.markdown(
    """
    - Incorporate **exogenous variables** such as market sentiment or volume indicators.
    - Explore **ensemble methods** combining statistical and deep learning models.
    - Extend forecasting to **probabilistic confidence intervals**.
    - Evaluate performance on **rolling-window validation**.
    """
)

# ============================================================
# FINAL CONCLUSION
# ============================================================

st.subheader("✅ Final Conclusion")

st.markdown(
    """
    This project demonstrates that no single model universally outperforms others
    across all cryptocurrencies. Deep learning models such as LSTM show superior
    performance for volatile assets, while classical models offer interpretability
    and stability. The results highlight the importance of **model selection based
    on data characteristics and forecasting horizon**.
    """
)
