import scipy.stats as stats
import pandas as pd


def fit_student_t(data: pd.Series):
    nu, loc, scale = stats.t.fit(data)
    t_dist = stats.t(df=nu, loc=loc, scale=scale)
    return t_dist


def fit_normal(data: pd.Series):
    loc, scale = stats.norm.fit(data)
    n_dist = stats.norm(loc=loc, scale=scale)
    return n_dist
