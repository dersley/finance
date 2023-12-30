import plotly.graph_objects as go
import datetime as dt
import pandas as pd
import numpy as np

from lib.ticker import Ticker
from lib.utils import fitting as fit, helper as help


def plot_returns_fit(ticker: Ticker):
    """
    Returns the fitted T distribution of the ticker along with its log return data.
    Use to assess quality of fit.
    """

    returns_data = ticker.get_log_returns()
    x = np.linspace(returns_data.min(), returns_data.max(), 1000)
    t_pdf = ticker.dist.pdf(x)

    # Create a Plotly figure
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x, y=t_pdf, mode="lines", name="Student T", line=dict(color="lime")
        )
    )

    # Add histogram for historical data
    fig.add_trace(
        go.Histogram(
            x=returns_data,
            histnorm="probability density",
            name="Historical",
            marker=dict(color="dodgerblue"),
        )
    )

    # Update the layout
    fig.update_layout(
        title=f"Historical Fit of Daily Log Returns: {ticker.code}",
        xaxis_title="Daily Log Returns (Closing Price)",
        yaxis_title="Density",
        showlegend=False,
        height=500,
    )

    return fig


def plot_log_returns(ticker: Ticker, start_date: dt.datetime):
    data = ticker.get_log_returns()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index, y=data.values, mode="lines", line=dict(color="dodgerblue")
        )
    )

    fig.update_layout(
        title="Daily Log Returns", xaxis_title="", yaxis_title="", height=500
    )

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
    bq, _, uq = help.calculate_percentiles(simdata, confidence=50, axis=0)

    # Create a continuous x-axis date range
    x_hist = historic_data.index
    x_sim = pd.date_range(
        start=x_hist[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="D"
    )

    # Create a Plotly figure
    fig = go.Figure()

    # Plot the historic data
    fig.add_trace(
        go.Scatter(
            x=x_hist,
            y=historic_data,
            mode="lines",
            name="Historic",
            line=dict(color="dodgerblue"),
        )
    )

    # Fill between the low and high percentiles
    fig.add_trace(
        go.Scatter(
            x=list(x_sim) + list(x_sim)[::-1],
            y=list(bq) + list(uq)[::-1],
            fill="toself",
            name="50% CI",
            marker=dict(color="lime"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(x_sim) + list(x_sim)[::-1],
            y=list(low) + list(high)[::-1],
            fill="toself",
            name="95% CI",
            marker=dict(color="lightgreen"),
        )
    )

    # Update the layout
    fig.update_layout(
        title=f"Simulated and Historic Asset Performance: {ticker.code}",
        xaxis_title="",
        yaxis_title="Balance $AUD",
        height=600,
    )

    return fig


def plot_ticker_autocorrelation(ticker: Ticker, max_lag=60):
    df = ticker.calculate_autocorrelation(max_lag=max_lag)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Lag"],
            y=df["Directional"],
            mode="lines",
            line=dict(color="Lime"),
            name="Directional",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Lag"],
            y=df["Non-Directional"],
            mode="lines",
            line=dict(color="Orange"),
            name="Non-Directional",
        )
    )
    fig.update_layout(
        xaxis_title="Lag (Days)",
        yaxis_title="Correlation Coefficient",
        title=f"Lag-wise Correlation: {ticker.code}",
        height=500,
    )

    return fig
