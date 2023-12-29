import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt

from .utils import fitting as fit, simulation as sim


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
        self.df = self.build_ticker_df()
        self.dist = self.fit_log_returns_dist()

    def build_ticker_df(self):
        df = self.yticker.history(
            start=self.start_date, end=dt.datetime.today(), interval="1d"
        )

        # Convert timestamp to standard timezone naive datetime
        df.index = df.index.to_pydatetime()
        df.index = df.index.tz_localize(None)

        return df

    def get_company_name(self):
        return self.yticker.info.get("longName")

    def get_current_price(self):
        return self.df["Close"].iloc[-1]

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

    def calculate_autocorrelation(self, max_lag=60) -> pd.DataFrame:
        log_returns = self.get_log_returns()
        correlations = sim.calculate_lag_correlations(log_returns, max_lag=max_lag)
        return correlations

    def simulate_returns(self, days: int, starting_balance: float, sims=1000):
        """
        Simulate returns for a given number of days from a starting balance

        Returns an array of daily balances of shape (sims, days).
        """

        log_returns = np.zeros((days, sims))
        for i in range(days):
            log_returns[i, :] = self.dist.rvs(size=sims)

        cum_returns = np.cumsum(log_returns, axis=0)
        balance = starting_balance * np.exp(cum_returns.T)

        return balance
