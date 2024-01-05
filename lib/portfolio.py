import numpy as np
import pandas as pd

from .ticker import Ticker
from .utils import simulation as sim


class Portfolio:
    """
    Collection of Ticker objects with associated holdings for each.

    Constructed with a holdings dict, where each key is a ticker code and each value is the units owned.

    The years argument sets the length of time to go back for historical analysis.
    """

    def __init__(self, holdings: dict[str, int], years: int, asx=True):
        self.holdings = holdings
        self.tickers = self.build_tickers_list(years, asx)
        self.log_returns_df = self.build_log_returns_df()
        self.corr_matrix = self.get_corr_matrix()
        self.annual_dists = self.get_annualized_return_dists()

    def build_tickers_list(self, years, asx):
        tickers = []
        for code in self.holdings:
            tickers.append(Ticker(code, years=years, asx=asx))
        return tickers

    def build_log_returns_df(self):
        data = {}
        for ticker in self.tickers:
            data[ticker.code] = ticker.get_log_returns()

        return pd.DataFrame(data)

    def get_ticker_codes(self):
        codes = []
        for ticker in self.tickers:
            codes.append(ticker.code)
        return codes

    def get_holdings_df(self):
        data = {
            "Code": [],
            "Company Name": [],
            "Share Price": [],
            "Units": [],
            "Value": [],
        }
        for ticker in self.tickers:
            price = ticker.get_current_price()
            units = self.get_units(ticker.code)

            data["Code"].append(ticker.code)
            data["Company Name"].append(ticker.get_company_name())
            data["Share Price"].append(price)
            data["Units"].append(units)
            data["Value"].append(round(units * price, 2))

        return pd.DataFrame(data)

    def get_starting_balance(self):
        balance = 0
        for ticker in self.tickers:
            price = ticker.get_current_price()
            units = self.get_units(ticker.code)
            balance += price * units
        return balance

    def get_corr_matrix(self):
        return self.log_returns_df.corr()

    def get_cov_matrix(self):
        return self.log_returns_df.cov()

    def get_units(self, ticker_code: str):
        return self.holdings[ticker_code]

    def get_annualized_return_dists(self) -> list:
        dists = []
        for ticker in self.tickers:
            dists.append(ticker.get_annualized_return_dist())
        return dists

    def get_annualized_returns(self) -> np.ndarray[float]:
        returns = []
        for ticker in self.tickers:
            returns.append(ticker.get_estimated_annualized_returns())
        return np.array(returns)

    def get_annualized_volatilities(self) -> np.ndarray[float]:
        volatilities = []
        for ticker in self.tickers:
            volatilities.append(ticker.get_estimated_annualized_returns())
        return np.array(volatilities)

    def simulate_correlated_returns(self, days: int, sims=1000):
        """
        Simulate correlated portfolio returns.

        Returns a 3D array of shape (days, num_tickers, sims)
        """
        num_tickers = len(self.tickers)
        log_returns = np.zeros((days, num_tickers, sims))
        for i in range(days):
            uniform_samples = sim.simulate_correlated_uniform_samples(
                num_elements=num_tickers,
                corr_matrix=self.get_corr_matrix(),
                sims=sims,
            )

            # Transpose to (num_tickers, sims)
            uniform_samples = uniform_samples.T

            for j, ticker in enumerate(self.tickers):
                log_returns[i, j, :] = ticker.dist.ppf(uniform_samples[j, :])

        return log_returns

    def simulate_correlated_annualized_returns(self, sims=1000):
        """
        Simulate correlated annualized portfolio returns for use in portfolio optimization.

        Returns a 2D array of shape (num_tickers, sims)
        """
        num_tickers = len(self.tickers)
        log_returns = np.zeros((num_tickers, sims))
        annual_dists = self.annual_dists

        uniform_samples = sim.simulate_correlated_uniform_samples(
            num_elements=num_tickers, corr_matrix=self.get_corr_matrix(), sims=sims
        )

        # Transpose to (num_tickers, sims)
        uniform_samples = uniform_samples.T

        for i, ticker in enumerate(self.tickers):
            log_returns[i, :] = annual_dists[i].ppf(uniform_samples[i, :])

        return log_returns

    def simulate_portfolio(self, days: int, sims=1000):
        num_tickers = len(self.tickers)
        log_returns = self.simulate_correlated_returns(days=days, sims=sims)

        starting_portfolio_balance = self.get_starting_balance()

        ticker_balance = np.zeros((num_tickers, days, sims))
        for i, ticker in enumerate(self.tickers):
            price = ticker.get_current_price()
            units = self.get_units(ticker.code)

            ticker_returns = log_returns[:, i, :]
            cum_returns = np.cumsum(ticker_returns, axis=0)

            starting_balance = price * units
            ticker_balance[i] = starting_balance * np.exp(cum_returns)

        portfolio_balance = np.sum(ticker_balance, axis=0)

        # Add starting balance to start of each simulation
        starting_balance_col = np.full(
            shape=sims, fill_value=starting_portfolio_balance
        )
        portfolio_balance = np.vstack((starting_balance_col, portfolio_balance))

        # Return the array in shape (sims, days)
        return portfolio_balance.T

    def simulate_portfolio_optimization(self, sims=10_000):
        num_tickers = len(self.tickers)
        weights = sim.generate_random_weights(num_tickers, sims=sims)
        returns = self.get_annualized_returns()
        volatilities = self.get_annualized_volatilities()
        corr_matrix = self.get_corr_matrix()

        # Upscale corr matrix by annualized volatilities
        cov_matrix = sim.convert_correlation_matrix(corr_matrix.values, volatilities)

        portfolio_simdata = np.zeros((sims, 2))
        for p in range(sims):
            portfolio_simdata[p] = sim.calculate_portfolio_performance(
                returns, weights[p], cov_matrix
            )

        df = pd.DataFrame(weights, columns=self.get_ticker_codes())
        df["Mean"] = portfolio_simdata[:, 0]
        df["Volatility"] = portfolio_simdata[:, 1]

        return df
