# app.py
import streamlit as st

# Main entry for Streamlit
st.set_page_config(
    page_title="Crypto Forecasting System",
    layout="wide"
)

def main():
    st.title("Crypto Forecasting System – AE2")

    st.markdown("""
    ### Welcome
    Use the navigation menu on the left to open the **Dashboard** page.

    Your dashboard file is located at:
    `streamlit/pages/1_Dashboard.py`

    Streamlit will automatically load and display that page.
    """)

    st.info("Go to the sidebar → Pages → 1_Dashboard")

if __name__ == "__main__":
    main()
