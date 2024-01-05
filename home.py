import streamlit as st
import pandas as pd
from lib.ticker import Ticker
from lib.portfolio import Portfolio
from app import dynamic_plots as plot

st.set_page_config(layout="wide")


def update_ticker(ticker_code, years, asx):
    if ticker_code:
        st.session_state.ticker = Ticker(code=ticker_code, years=years, asx=asx)
        st.session_state.company_name = st.session_state.ticker.get_company_name()


def update_portfolio(portfolio_df, years, asx):
    holdings = {}
    for i, row in portfolio_df.iterrows():
        holdings[row["Code"]] = row["Units"]

    st.session_state.portfolio = Portfolio(holdings, years=years, asx=asx)


def add_tickers_to_portfolio():
    new_ticker_list = [item.strip() for item in new_tickers.split(",")]
    new_ticker_list = list(set(new_ticker_list))
    updated_ticker_list = set(new_ticker_list).union(set(st.session_state.ticker_list))

    st.session_state.ticker_list = updated_ticker_list


def remove_tickers_from_portfolio():
    tickers_to_remove = [item.strip() for item in new_tickers.split(",")]

    st.session_state.ticker_list = [
        ticker
        for ticker in st.session_state.ticker_list
        if ticker not in tickers_to_remove
    ]


if "ticker_list" not in st.session_state:
    st.session_state.ticker_list = []

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
        st.divider()

        ticker_list = st.session_state.ticker_list
        st.dataframe(ticker_list, hide_index=True, use_container_width=True)

        new_tickers = st.text_input("Add Ticker")
        col1, col2 = st.columns(2)
        add_ticker_button = col1.button(
            "Add Ticker(s)", on_click=add_tickers_to_portfolio
        )
        remove_ticker_button = col2.button(
            "Remove Ticker(s)", on_click=remove_tickers_from_portfolio
        )

    st.title("Portfolio Analysis")
    st.divider()

    if not st.session_state.ticker_list:
        st.stop()

    holdings = {key: 100 for key in st.session_state.ticker_list}

    simulate_portfolio = st.button("Optimize Portfolio")
    if simulate_portfolio:
        st.session_state.portfolio = Portfolio(holdings=holdings, years=years, asx=asx)

        seg1, seg2 = st.columns(2)

        heatmap = plot.plot_portfolio_corr_heatmap(st.session_state.portfolio)
        portfolio_optimization_plot = plot.plot_portfolio_optimization(
            st.session_state.portfolio
        )

        seg1.plotly_chart(heatmap, use_container_width=True)
        seg2.plotly_chart(portfolio_optimization_plot, use_container_width=True)
