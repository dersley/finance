import streamlit as st
from lib.ticker import Ticker
from app import dynamic_plots as plot

def update_ticker(ticker_code, years, asx):
    if ticker_code:
        st.session_state.ticker = Ticker(code=ticker_code, years=years, asx=asx)
        st.session_state.company_name = st.session_state.ticker.get_company_name()

st.set_page_config(layout="wide")

seg1, seg2 = st.columns([0.25, 0.75])

with seg1:
    if "company_name" in st.session_state:
        st.title(st.session_state.company_name)

    ticker_update_form = st.form("ticker_update_form")
    with ticker_update_form:
        market = st.radio("Market", options=["ASX", "NASDAQ"], horizontal=True)
        asx = market == "ASX"

        ticker_code = st.text_input("Ticker Code")
        years = st.number_input("Years historic data", min_value=1, max_value=10, value=5)
        forecast_days = st.number_input("Forecast Days", min_value=10, max_value=1000, value=100)

        # Update ticker on any change
        if st.form_submit_button('Update Ticker'):
            update_ticker(ticker_code, years, asx)

# Initialize the ticker in session state if it doesn't exist
if "ticker" not in st.session_state:
    st.info("Enter a ticker code to get started.")
    st.stop()

with seg2:
    col1, col2 = st.columns(2)
    fig1 = plot.plot_returns_fit(st.session_state.ticker)
    col1.plotly_chart(fig1, use_container_width=True)

    fig2 = plot.plot_ticker_autocorrelation(st.session_state.ticker, max_lag=100)
    col2.plotly_chart(fig2, use_container_width=True)

fig3 = plot.plot_simulated_balance(st.session_state.ticker, st.session_state.ticker.start_date, forecast_days)
st.plotly_chart(fig3, use_container_width=True)
