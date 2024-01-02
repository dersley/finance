import matplotlib.pyplot as plt
import seaborn as sns

from app import dynamic_plots as plot
from lib.utils import simulation as sim, helper as help
from lib.ticker import Ticker
from lib.portfolio import Portfolio


def main():
    years = 10
    asx = True

    holdings_dict = {
        "WDS": 100,
        "QUAL": 100,
        "CBA": 100,
        "QAN": 333,
    }

    portfolio = Portfolio(holdings=holdings_dict, years=years, asx=asx)
    mean, volatility, weights = portfolio.simulate_portfolio_optimization(
        portfolios=10_000, sims=1000
    )

    fig = plot.plot_portfolio_optimization(portfolio, mean, volatility)
    fig.show()


if __name__ == "__main__":
    main()
