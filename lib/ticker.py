import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt

from .utils import fitting as fit


class Ticker:
    """
    A stock price history from yahoo finance and associated statistical analysis.

    When constructed, the ticker price data is loaded and a distribution is fit to log returns over the chosen date period.
    The timeframe of data is a given number of years ago to now.

    If asx is True (Australian stock exchange), a ".AX" is appended to the ticker code.
    """

    def __init__(self, code: str, years: int, asx=True):
        self.code = code

        if asx:
            code += ".AX"

        self.start_date = dt.datetime.today() - dt.timedelta(days=int(365 * years))
        self.yticker = yf.Ticker(ticker=code)
        self.df = self.yticker.history(
            start=self.start_date, end=dt.datetime.today(), interval="1d"
        )

        self.dist = self.fit_log_returns_dist()

    def get_log_returns(self):
        """
        Return a dataframe of log returns based on daily closing price
        """
        returns_data = np.log(self.df["Close"] / self.df["Close"].shift(1))
        returns_data = returns_data.dropna()
        return returns_data

    def fit_log_returns_dist(self):
        """
        Fits a students T distribution to the log returns data
        """
        returns_data = self.get_log_returns()
        return fit.fit_student_t(returns_data)
