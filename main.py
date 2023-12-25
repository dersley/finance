import matplotlib.pyplot as plt
import datetime as dt

from lib.portfolio import Portfolio
from app import static_plots as plot



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
    }

    portfolio = Portfolio(holdings_dict, years=years, asx=True)

    test_ticker = portfolio.tickers[4]

    fig = plot.plot_returns_fit(test_ticker)
    plt.show()

    start_date = dt.datetime(2021, 1, 1)
    fig = plot.plot_simulated_balance(
        ticker=test_ticker,
        start_date=start_date,
        forecast_days=365,
        sims=2500
    )
    plt.show()

if __name__ == "__main__":
    main()
