import streamlit as st
import pandas as pd
from lib.ticker import Ticker
from lib.portfolio import Portfolio
from app import dynamic_plots as plot


def update_ticker(ticker_code, years, asx):
    if ticker_code:
        st.session_state.ticker = Ticker(code=ticker_code, years=years, asx=asx)
        st.session_state.company_name = st.session_state.ticker.get_company_name()


def update_portfolio(portfolio_df, years, asx):
    holdings = {}
    for i, row in portfolio_df.iterrows():
        holdings[row["Code"]] = row["Units"]

    st.session_state.portfolio = Portfolio(holdings, years=years, asx=asx)


st.set_page_config(layout="wide")

st.session_state.page = st.sidebar.radio(
    label="Page", options=["Individual", "Portfolio"], horizontal=True
)
st.sidebar.divider()

if st.session_state.page == "Individual":
    if "company_name" in st.session_state:
        st.title(st.session_state.company_name)
        st.divider()

    seg1, seg2, seg3 = st.columns([2, 3, 2])

    ticker_update_form = st.sidebar.form("ticker_update_form")
    with ticker_update_form:
        market = st.radio("Market", options=["ASX", "NASDAQ"], horizontal=True)
        asx = market == "ASX"

        ticker_code = st.text_input("Ticker Code")
        years = st.number_input(
            "Years historic data", min_value=1, max_value=25, value=5
        )
        forecast_days = st.number_input(
            "Forecast Days", min_value=10, max_value=1000, value=100
        )

        # Update ticker on any change
        if st.form_submit_button("Submit"):
            update_ticker(ticker_code, years, asx)

    if "ticker" not in st.session_state:
        st.info("Enter a ticker code to get started.")
        st.stop()

    with seg1:
        fig1 = plot.plot_returns_fit(st.session_state.ticker)
        st.plotly_chart(fig1, use_container_width=True)

    with seg2:
        fig2 = plot.plot_log_returns(
            st.session_state.ticker, start_date=st.session_state.ticker.start_date
        )
        st.plotly_chart(fig2, use_container_width=True)

    with seg3:
        fig3 = plot.plot_ticker_autocorrelation(st.session_state.ticker, max_lag=100)
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    fig3 = plot.plot_simulated_balance(
        st.session_state.ticker, st.session_state.ticker.start_date, forecast_days
    )
    st.plotly_chart(fig3, use_container_width=True)

elif st.session_state.page == "Portfolio":
    with st.sidebar:
        market = st.radio("Market", options=["ASX", "NASDAQ"], horizontal=True)
        asx = market == "ASX"
        years = st.number_input("Years", min_value=1, max_value=25, value=3)

    st.title("Portfolio Analysis")

    if "portfolio" not in st.session_state:
        holdings_dict = {
            "QUAL": 100,
            "STO": 4990,
            "CBA": 100,
            "VAS": 100,
            "QAN": 333,
        }

        portfolio_df = pd.DataFrame(
            {
                "Code": [],
                "Company Name": [],
                "Share Price": [],
                "Units": [],
                "Value": [],
            }
        )
    else:
        holdings_dict = st.session_state.portfolio.holdings
        portfolio_df = st.session_state.portfolio.get_holdings_df()

    portfolio_expander = st.expander("Portfolio", expanded=True)
    with portfolio_expander:
        new_df = st.data_editor(portfolio_df, use_container_width=True, hide_index=True)
        update_portfolio_button = st.button("Update")
        if update_portfolio_button:
            update_portfolio(portfolio_df, years, asx)

    st.divider()

    st.session_state.portfolio = Portfolio(holdings=holdings_dict, years=years, asx=asx)
    seg1, seg2 = st.columns(2)

    fig1 = plot.plot_portfolio_corr_heatmap(st.session_state.portfolio)
    seg1.plotly_chart(fig1, use_container_width=True)

    fig2 = plot.plot_simulated_portfolio(
        portfolio=st.session_state.portfolio, forecast_days=365, sims=1000
    )
    seg2.plotly_chart(fig2, use_container_width=True)
