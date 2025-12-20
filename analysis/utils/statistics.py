"""
Statistical analysis utilities for analysis scripts.

Provides functions for finding best values, statistical significance testing,
and computing relative/absolute changes.
"""

from typing import List, Optional
from scipy import stats


def find_best_in_column(values: List) -> int:
    """
    Find index of best (maximum) value in a list, ignoring None.

    Handles both simple values and tuples (mean, std, raw_values).

    Args:
        values: List of values or tuples

    Returns:
        Index of best value, or -1 if no valid values

    Example:
        >>> find_best_in_column([0.10, 0.30, 0.20])
        1
        >>> find_best_in_column([(0.10, 0.01, []), (0.30, 0.02, []), None])
        1
    """
    valid_values = []
    for i, v in enumerate(values):
        if v is not None:
            # Extract mean from tuple if needed
            if isinstance(v, tuple):
                mean = v[0]  # First element is always mean
                if mean is not None:
                    valid_values.append((i, mean))
            else:
                valid_values.append((i, v))

    if not valid_values:
        return -1
    return max(valid_values, key=lambda x: x[1])[0]


def is_significantly_better(
    best_values: Optional[List[float]],
    other_values: Optional[List[float]],
    alpha: float = 0.05
) -> bool:
    """
    Test if best_values is significantly better than other_values using t-test.

    Uses paired t-test if sample sizes match, independent t-test otherwise.

    Args:
        best_values: Raw values for the best method
        other_values: Raw values for the comparison method
        alpha: Significance level (default 0.05)

    Returns:
        True if best_values is significantly better (p < alpha), False otherwise

    Example:
        >>> best = [0.50, 0.52, 0.51, 0.53, 0.49]
        >>> worse = [0.30, 0.32, 0.31, 0.33, 0.29]
        >>> is_significantly_better(best, worse)
        True
    """
    if best_values is None or other_values is None:
        return False

    if len(best_values) == 0 or len(other_values) == 0:
        return False

    # If either has only one value, we can't do a proper test
    if len(best_values) == 1 or len(other_values) == 1:
        return False

    # Ensure same number of samples for paired t-test
    if len(best_values) != len(other_values):
        # Fall back to independent t-test if sample sizes differ
        t_stat, p_value = stats.ttest_ind(best_values, other_values, alternative='greater')
    else:
        # Use paired t-test (assumes same seeds across methods)
        t_stat, p_value = stats.ttest_rel(best_values, other_values, alternative='greater')

    return p_value < alpha


def find_best_with_significance(
    values: List,
    alpha: float = 0.05
) -> int:
    """
    Find index of best value that is statistically significantly better than all others.

    Returns -1 if no value is significantly better than all others.

    Args:
        values: List of tuples (mean, std, raw_values)
        alpha: Significance level (default 0.05)

    Returns:
        Index of significantly best value, or -1

    Example:
        >>> values = [
        ...     (0.30, 0.01, [0.29, 0.30, 0.31]),
        ...     (0.50, 0.01, [0.49, 0.50, 0.51]),  # Significantly better
        ... ]
        >>> find_best_with_significance(values)
        1
    """
    # Find the index with maximum mean
    best_idx = find_best_in_column(values)

    if best_idx == -1:
        return -1

    best_value = values[best_idx]
    if best_value is None or not isinstance(best_value, tuple) or len(best_value) < 3:
        return -1

    best_mean, _, best_raw_values = best_value

    # Check if significantly better than all other methods
    for i, other_value in enumerate(values):
        if i == best_idx:
            continue

        if other_value is None:
            continue

        if not isinstance(other_value, tuple) or len(other_value) < 3:
            continue

        other_mean, _, other_raw_values = other_value

        if other_mean is None:
            continue

        # Test if best is significantly better than this method
        if not is_significantly_better(best_raw_values, other_raw_values, alpha):
            # Not significantly better than at least one other method
            return -1

    # Significantly better than all other methods
    return best_idx


def compute_relative_change(
    baseline: Optional[float],
    comparison: Optional[float]
) -> Optional[float]:
    """
    Compute relative percentage change: (comparison - baseline) / baseline * 100.

    Args:
        baseline: Baseline value
        comparison: Comparison value

    Returns:
        Relative change in percentage, or None if invalid

    Example:
        >>> compute_relative_change(0.50, 0.60)
        20.0
        >>> compute_relative_change(0.50, 0.40)
        -20.0
    """
    if baseline is None or comparison is None or baseline == 0:
        return None
    return ((comparison - baseline) / baseline) * 100


def compute_absolute_change(
    baseline: Optional[float],
    comparison: Optional[float]
) -> Optional[float]:
    """
    Compute absolute change: comparison - baseline.

    Args:
        baseline: Baseline value
        comparison: Comparison value

    Returns:
        Absolute change, or None if invalid

    Example:
        >>> compute_absolute_change(0.50, 0.60)
        0.10
        >>> compute_absolute_change(0.50, 0.40)
        -0.10
    """
    if baseline is None or comparison is None:
        return None
    return comparison - baseline


def compute_pvalue(
    values_a: Optional[List[float]],
    values_b: Optional[List[float]],
    alternative: str = 'two-sided'
) -> Optional[float]:
    """
    Compute p-value for the difference between two sets of values using t-test.

    Uses paired t-test if sample sizes match, independent t-test otherwise.

    Args:
        values_a: Raw values for method A
        values_b: Raw values for method B
        alternative: 'two-sided', 'greater' (A > B), or 'less' (A < B)

    Returns:
        p-value, or None if test cannot be performed

    Example:
        >>> a = [0.50, 0.52, 0.51, 0.53, 0.49]
        >>> b = [0.30, 0.32, 0.31, 0.33, 0.29]
        >>> p = compute_pvalue(a, b, alternative='greater')
        >>> p < 0.05
        True
    """
    if values_a is None or values_b is None:
        return None

    if len(values_a) == 0 or len(values_b) == 0:
        return None

    # If either has only one value, we can't do a proper test
    if len(values_a) == 1 or len(values_b) == 1:
        return None

    # Ensure same number of samples for paired t-test
    if len(values_a) != len(values_b):
        # Fall back to independent t-test if sample sizes differ
        _, p_value = stats.ttest_ind(values_a, values_b, alternative=alternative)
    else:
        # Use paired t-test (assumes same seeds across methods)
        _, p_value = stats.ttest_rel(values_a, values_b, alternative=alternative)

    return p_value
