import numpy as np
import matplotlib.pyplot as plt

from lib.ticker import Ticker
from lib.portfolio import Portfolio
from lib import ticker_groups as groups
from app import static_plots as plot


def main():
    years = 3

    holdings_dict = {"WDS": 10_000, "STO": 10_000, "SEN": 10_000}

    portfolio = Portfolio(holdings_dict, years=years, asx=True)

    print(portfolio.corr_matrix)


if __name__ == "__main__":
    main()
