import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from lib.ticker import Ticker
from lib.utils import fitting as fit


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

    fig, ax = plt.subplots(figsize=(16, 9))

    sns.lineplot(x=x, y=t_pdf, ax=ax, color="red", label="Student T")
    sns.lineplot(x=x, y=n_pdf, ax=ax, color="lime", label="Normal")
    sns.histplot(returns_data, stat="density", ax=ax, label="Historical")

    ax.set_title(f"Historical fit of log returns: {ticker.code}")
    ax.set_xlabel("Daily Log Returns (Closing Price)")
    ax.legend()

    return fig
