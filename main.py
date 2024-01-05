import matplotlib.pyplot as plt
import seaborn as sns

from app import dynamic_plots as plot
from lib.ticker import Ticker
from lib.portfolio import Portfolio


def main():
    years = 10
    asx = True

    holdings_dict = {
        "VAS": 100,
        "QUAL": 100,
        "QAN": 100,
        "STO": 100,
        "CHN": 100
    }

    portfolio = Portfolio(holdings=holdings_dict, years=years, asx=asx)

    fig = plot.plot_portfolio_optimization(portfolio)
    fig.show()


if __name__ == "__main__":
    main()
