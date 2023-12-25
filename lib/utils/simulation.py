import numpy as np
import scipy.stats as stats


def simulate_correlated_uniform_samples(num_elements: int, corr_matrix, sims=10_000):
    means = np.zeros(num_elements)
    mvn_samples = np.random.multivariate_normal(means, cov=corr_matrix, size=sims)
    uniform_samples = stats.norm.cdf(mvn_samples)
    return uniform_samples
