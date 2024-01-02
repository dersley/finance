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

    def get_units(self, ticker_code: str):
        return self.holdings[ticker_code]

    def get_annualized_return_dists(self) -> list:
        dists = []
        for ticker in self.tickers:
            dists.append(ticker.get_annualized_return_dist())
        return dists

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

    def simulate_portfolio_optimization(
        self, portfolios=1000, sims=1000
    ) -> tuple[np.ndarray]:
        num_tickers = len(self.tickers)
        weights = sim.generate_random_weights(num_tickers, sims=portfolios)
        returns_dists = self.get_annualized_return_dists()

        returns = np.zeros((portfolios, num_tickers, sims))
        for p in range(portfolios):
            for t, ticker in enumerate(self.tickers):
                weight = weights[p, t]
                samples = returns_dists[t].rvs(size=sims)
                returns[p, t, :] = samples * weight

        total_returns = np.sum(returns, axis=1)
        mean = np.mean(total_returns, axis=1)
        volatility = np.std(total_returns, axis=1)

        return mean, volatility, weights
