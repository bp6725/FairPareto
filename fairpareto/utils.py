"""
Utility functions for fairpareto package.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_to_sets(df, test_size=0.3):
    """
    Split dataset into train, test, and validation sets with stratification.

    This function splits the data while maintaining the distribution of
    sensitive attribute and label combinations across all splits.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with 'attr' (sensitive attribute) and 'label' columns.
    test_size : float, default=0.3
        Proportion of data to use for test+validation (will be split equally).

    Returns
    -------
    tuple
        (train, test, val) - Three dataframes with stratified splits.
    """
    # Create stratification column combining sensitive attribute and label
    df["stratify_column"] = df["attr"].astype(str) + "_" + df["label"].astype(str)

    # First split: train vs (test + validation)
    train, temp = train_test_split(
        df, test_size=test_size, stratify=df["stratify_column"], random_state=42
    )

    # Second split: test vs validation
    test, val = train_test_split(
        temp, test_size=0.5, stratify=temp["stratify_column"], random_state=42
    )

    # Clean up datasets
    for dataset in [train, test, val]:
        dataset.drop("stratify_column", axis=1, inplace=True)
        dataset.reset_index(drop=True, inplace=True)

    return train, test, val


def equal_frequency_binning(c_over_samples, n_bins):
    """
    Apply equal frequency binning to probability values.

    This function creates bins such that each bin contains approximately
    the same number of samples.

    Parameters
    ----------
    c_over_samples : array-like
        Probability values to bin.
    n_bins : int
        Number of bins to create.

    Returns
    -------
    tuple
        (f_bined, bins_centers, n_samples) where:
        - f_bined: normalized frequencies for each bin
        - bins_centers: center points of each bin
        - n_samples: total number of samples
    """
    # Sort the data
    sorted_data = np.sort(c_over_samples)

    # Calculate the number of elements per bin (ideally)
    n_elements = len(sorted_data)
    elements_per_bin = n_elements // n_bins

    # Create the bins with equal frequency
    bins = np.zeros(n_bins + 1)
    bins[0] = sorted_data[0] - 1e-10  # Start slightly below the minimum value

    for i in range(1, n_bins):
        bin_edge_idx = i * elements_per_bin
        bins[i] = sorted_data[bin_edge_idx]

    bins[-1] = sorted_data[-1] + 1e-10  # End slightly above the maximum value

    # Calculate bin centers
    bins_centers = (bins[:-1] + bins[1:]) / 2

    # Use np.histogram with these custom bins
    f_bined, _ = np.histogram(c_over_samples, bins)

    # Normalize frequencies
    f_bined = f_bined / sum(f_bined)

    return f_bined.reshape(-1, 1), bins_centers.reshape(-1, 1), len(c_over_samples)


def validate_binary_labels(y):
    """
    Validate that labels are binary (0 and 1).

    Parameters
    ----------
    y : array-like
        Label array to validate.

    Raises
    ------
    ValueError
        If labels are not binary or contain invalid values.
    """
    unique_labels = np.unique(y)
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"Labels must be binary (0 and 1). Found: {unique_labels}")

    if len(unique_labels) < 2:
        raise ValueError(f"Both classes (0 and 1) must be present. Found: {unique_labels}")


def validate_sensitive_attribute(s):
    """
    Validate that sensitive attribute is binary (0 and 1).

    Parameters
    ----------
    s : array-like
        Sensitive attribute array to validate.

    Raises
    ------
    ValueError
        If sensitive attribute is not binary or contains invalid values.
    """
    unique_attrs = np.unique(s)
    if not set(unique_attrs).issubset({0, 1}):
        raise ValueError(f"Sensitive attribute must be binary (0 and 1). Found: {unique_attrs}")

    if len(unique_attrs) < 2:
        raise ValueError(f"Both sensitive groups (0 and 1) must be present. Found: {unique_attrs}")


def check_data_consistency(X, y, sensitive_column=None):
    """
    Check data consistency and validate inputs.

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Input features.
    y : array-like
        Target labels.
    sensitive_column : str, optional
        Name of sensitive attribute column in X.

    Raises
    ------
    ValueError
        If data is inconsistent or invalid.
    """
    # Check shapes match
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length. Got X: {len(X)}, y: {len(y)}")

    # Validate labels
    validate_binary_labels(y)

    # Validate sensitive attribute if provided
    if sensitive_column is not None:
        if isinstance(X, pd.DataFrame):
            if sensitive_column not in X.columns:
                raise ValueError(f"Sensitive column '{sensitive_column}' not found in X")
            validate_sensitive_attribute(X[sensitive_column])
        else:
            raise ValueError("X must be a DataFrame when sensitive_column is specified")

    # Check for missing values
    if isinstance(X, pd.DataFrame):
        if X.isnull().any().any():
            raise ValueError("X contains missing values")
    elif isinstance(X, np.ndarray):
        if np.isnan(X).any():
            raise ValueError("X contains missing values")

    if np.isnan(y).any():
        raise ValueError("y contains missing values")


def compute_group_statistics(y, s):
    """
    Compute basic statistics for each sensitive group.

    Parameters
    ----------
    y : array-like
        Target labels.
    s : array-like
        Sensitive attribute values.

    Returns
    -------
    dict
        Dictionary containing group statistics.
    """
    stats = {}

    for group in [0, 1]:
        mask = s == group
        group_y = y[mask]

        stats[f'group_{group}'] = {
            'size': len(group_y),
            'positive_rate': np.mean(group_y),
            'negative_rate': 1 - np.mean(group_y)
        }

    # Overall statistics
    stats['overall'] = {
        'size': len(y),
        'positive_rate': np.mean(y),
        'group_0_proportion': np.mean(s == 0),
        'group_1_proportion': np.mean(s == 1)
    }

    return stats


def format_pareto_results(pareto_front, decimals=4):
    """
    Format Pareto front results for display.

    Parameters
    ----------
    pareto_front : dict
        Dictionary mapping gamma values to accuracy values.
    decimals : int, default=4
        Number of decimal places for rounding.

    Returns
    -------
    pandas.DataFrame
        Formatted results as a DataFrame.
    """
    df = pd.DataFrame(list(pareto_front.items()),
                      columns=['Fairness_Level_Gamma', 'Optimal_Accuracy'])

    df = df.round(decimals)
    df = df.sort_values('Fairness_Level_Gamma').reset_index(drop=True)

    return df