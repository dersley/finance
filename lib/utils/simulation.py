import pandas as pd
import numpy as np
import scipy.stats as stats


def simulate_correlated_uniform_samples(num_elements: int, corr_matrix, sims=1000):
    means = np.zeros(num_elements)
    mvn_samples = np.random.multivariate_normal(means, cov=corr_matrix, size=sims)
    uniform_samples = stats.norm.cdf(mvn_samples)
    return uniform_samples


def calculate_lag_correlations(series, max_lag):
    """
    Calculate both directional and non-directional correlations for a range of lags in a time series.

    Returns:
    pandas.DataFrame: Dataframe containing both types of correlations for each lag.
    """
    directional_correlations = {}
    non_directional_correlations = {}

    # Directional correlation
    for lag in range(max_lag + 1):
        lagged_series = series.shift(lag)
        correlation = series.corr(lagged_series)
        directional_correlations[lag] = correlation

    # Non-directional correlation
    abs_series = series.abs()
    for lag in range(max_lag + 1):
        lagged_series = abs_series.shift(lag)
        correlation = abs_series.corr(lagged_series)
        non_directional_correlations[lag] = correlation

    # Combining into a DataFrame
    correlations_df = pd.DataFrame(
        {
            "Lag": list(range(max_lag + 1)),
            "Directional": list(directional_correlations.values()),
            "Non-Directional": list(non_directional_correlations.values()),
        }
    )

    return correlations_df


def convert_correlation_matrix(corr_matrix, volatilities) -> np.ndarray:
    """Upscale a correlation matrix to covariance using a set of volatilities."""

    n = len(volatilities)
    covariance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            covariance_matrix[i, j] = corr_matrix[i, j] * volatilities[i] * volatilities[j]
    return covariance_matrix


def calculate_portfolio_performance(returns, weights, cov_matrix) -> tuple[float, float]:
    portfolio_return = np.dot(weights, returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility


def generate_random_weights(num_securities, sims=1000):
    return np.random.dirichlet(alpha=np.ones(num_securities), size=sims)
