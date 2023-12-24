import numpy as np
import pandas as pd
import datetime as dt

from .ticker import Ticker


class Portfolio:
    """
    Collection of Ticker objects with associated holdings for each.

    Constructed with a holdings dict, where each key is a ticker code and each value is holdings in $AUD.

    The years argument sets the length of time to go back for historical analysis.
    """

    def __init__(self, holdings: dict[str, float], years: int, asx=True):
        self.holdings = holdings
        self.tickers = self.build_tickers_list(years, asx)
        self.log_returns_df = self.build_log_returns_df()
        self.corr_matrix = self.get_corr_matrix()

    def build_tickers_list(self, years, asx):
        tickers = []
        for code in self.holdings:
            tickers.append(Ticker(code, years=years, asx=asx))
        return tickers

    def build_log_returns_df(self):
        data = {}
        for ticker in self.tickers:
            data[ticker] = ticker.get_log_returns()

        return pd.DataFrame(data)

    def get_corr_matrix(self):
        return self.log_returns_df.corr()
