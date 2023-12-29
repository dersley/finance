import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import datetime as dt

from lib.ticker import Ticker
from lib.portfolio import Portfolio
from lib.utils import fitting as fit
from lib.utils import helper as help
from app import plot_styles as ps


def plot_returns_fit(ticker: Ticker):
    """
    Returns the fitted T distribution of the ticker along with its log return data.
    Use to assess quality of fit.
    """

    returns_data = ticker.get_log_returns()

    x_range = [returns_data.min(), returns_data.max()]
    x = np.linspace(x_range[0], x_range[1], 1000)

    n_dist = fit.fit_normal(returns_data)
    n_pdf = n_dist.pdf(x)
    t_pdf = ticker.dist.pdf(x)

    fig, ax = plt.subplots(figsize=ps.LONGPLOT)

    sns.lineplot(x=x, y=t_pdf, ax=ax, color="red", label="Student T")
    sns.lineplot(x=x, y=n_pdf, ax=ax, color="lime", label="Normal")
    sns.histplot(returns_data, stat="density", ax=ax, label="Historical")

    ax.set_title(f"Historical fit of log returns: {ticker.code}")
    ax.set_xlabel("Daily Log Returns (Closing Price)")
    ax.legend()

    return fig


def plot_simulated_balance(
    ticker: Ticker,
    start_date: dt.datetime,
    forecast_days: int,
    starting_balance=None,
    sims=1000,
):
    if starting_balance is None:
        starting_balance = ticker.get_current_price()

    historic_data = ticker.df["Close"][ticker.df.index >= start_date]

    # Simulate the future data
    simdata = ticker.simulate_returns(forecast_days, starting_balance, sims=sims)
    low, mid, high = help.calculate_percentiles(simdata, confidence=95, axis=0)
    p90, _, p10 = help.calculate_percentiles(simdata, confidence=80, axis=0)
    p75, _, p25 = help.calculate_percentiles(simdata, confidence=50, axis=0)

    fig, ax = plt.subplots(figsize=ps.LONGPLOT)

    # Create a continuous x-axis date range
    x_hist = historic_data.index
    x_sim = pd.date_range(
        start=x_hist[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="D"
    )

    # Plot the historic data
    sns.lineplot(x=x_hist, y=historic_data, ax=ax, label="Historic Data")

    # Plot the median simulation
    sns.lineplot(x=x_sim, y=mid, ax=ax, label="Median Performance")

    # Fill between the low and high percentiles
    ax.fill_between(x=x_sim, y1=low, y2=high, alpha=0.1, color="blue")
    ax.fill_between(x=x_sim, y1=p90, y2=p10, alpha=0.1, color="blue")
    ax.fill_between(x=x_sim, y1=p75, y2=p25, alpha=0.1, color="blue")

    ax.set_ylabel("Balance $AUD")
    ax.set_xlabel("")
    ax.set_title(f"Simulated and Historic Asset Performance: {ticker.code}")
    ax.legend()

    return fig


def plot_correlation_matrix(portfolio: Portfolio):
    corr_matrix = portfolio.get_corr_matrix()
    fig = sns.heatmap(
        corr_matrix,
        annot=True,
    )
    return fig
