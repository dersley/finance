import pandas as pd
import numpy as np
import scipy.stats as stats


def simulate_correlated_uniform_samples(num_elements: int, corr_matrix, sims=10_000):
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
    correlations_df = pd.DataFrame({
        "Lag": list(range(max_lag + 1)),
        "Directional": list(directional_correlations.values()),
        "Non-Directional": list(non_directional_correlations.values())
    })

    return correlations_df
