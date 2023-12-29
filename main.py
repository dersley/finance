import matplotlib.pyplot as plt
import datetime as dt

from lib.portfolio import Portfolio
from lib.ticker import Ticker
from lib.utils import helper as help
from app import dynamic_plots as plot


def main():
    years = 10

    holdings_dict = {
        "WDS": 10_000,
        "STO": 10_000,
        "SEN": 10_000,
        "CHN": 10_000,
        "TLS": 10_000,
        "CBA": 10_000,
        "GOVT": 10_000,
        "BEAR": 10_000,
    }

    portfolio = Portfolio(holdings_dict, years=years, asx=True)
    test_ticker = Ticker("BEAR", years, asx=True)
    start_date = dt.datetime(2020, 1, 1)

    fig = plot.plot_simulated_balance(test_ticker, start_date, 100)
    fig.show()

if __name__ == "__main__":
    main()
