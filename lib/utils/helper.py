import numpy as np
from prettytable import PrettyTable


def print_correlation_matrix(corr_matrix):
    """
    Prints a given correlation matrix (NumPy array) in a neatly formatted table.

    :param corr_matrix: NumPy array representing the correlation matrix.
    """
    if not isinstance(corr_matrix, np.ndarray):
        raise ValueError("The correlation matrix must be a NumPy array.")

    # Create a PrettyTable
    table = PrettyTable()

    # Add column names (First column is empty for the row labels)
    num_cols = corr_matrix.shape[1]
    table.field_names = [""] + [f"Col {i}" for i in range(num_cols)]

    # Add rows to the table
    for i, row in enumerate(corr_matrix):
        table.add_row([f"Row {i}"] + list(row))

    # Print the table
    print(table)


def calculate_percentiles(data, axis=0, confidence=95):
    percentiles = [(50 - confidence / 2), 50, (50 + confidence / 2)]
    return np.percentile(data, q=percentiles, axis=axis)
