"""
Metric extraction utilities for analysis scripts.

Provides functions for extracting metrics from wandb run history,
handling seeds, and computing statistics across multiple runs.
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple


def get_metric_at_iteration(
    history: List[Dict],
    metric_name: str,
    iteration: int
) -> Optional[float]:
    """
    Extract metric value at a specific iteration from run history using direct indexing.

    Args:
        history: List of history dictionaries from wandb run
        metric_name: Name of metric to extract (e.g., 'DAFNY2VERUS/pass@1')
        iteration: Array index to extract (e.g., 0 for iter1, 1 for iter2, 5 for iter5)

    Returns:
        Metric value at specified iteration index, or None if not found

    Example:
        >>> history = [
        ...     {'DAFNY2VERUS/pass@1': 0.10},
        ...     {'DAFNY2VERUS/pass@1': 0.12},
        ...     {'DAFNY2VERUS/pass@1': 0.15},
        ... ]
        >>> get_metric_at_iteration(history, 'DAFNY2VERUS/pass@1', 0)
        0.10
    """
    # Bounds check - make sure iteration index exists in history
    if iteration >= len(history) or iteration < 0:
        return None

    row = history[iteration]
    if metric_name in row and row[metric_name] is not None:
        # Handle NaN values stored as strings
        if isinstance(row[metric_name], str) and row[metric_name] == 'NaN':
            return None
        return row[metric_name]
    return None


def get_final_metric_value(history: List[Dict], metric_name: str) -> Optional[float]:
    """
    Extract the final (last) value for a given metric from run history.
    DEPRECATED: Use get_metric_at_iteration for iteration-specific extraction.

    Args:
        history: List of history dictionaries from wandb run
        metric_name: Name of metric to extract (e.g., 'DAFNY2VERUS/pass@1')

    Returns:
        Final value of the metric, or None if not found

    Example:
        >>> history = [
        ...     {'_step': 0, 'DAFNY2VERUS/pass@1': 0.10},
        ...     {'_step': 1, 'DAFNY2VERUS/pass@1': 0.15},
        ... ]
        >>> get_final_metric_value(history, 'DAFNY2VERUS/pass@1')
        0.15
    """
    values = []
    for row in history:
        if metric_name in row and row[metric_name] is not None:
            # Handle NaN values stored as strings
            if isinstance(row[metric_name], str) and row[metric_name] == 'NaN':
                continue
            values.append(row[metric_name])

    if values:
        return values[-1]  # Return last value
    return None


def get_base_run_name(run_name: str) -> str:
    """
    Extract base run name by removing seed suffix.

    Args:
        run_name: Full run name (e.g., 'SIMPLe-10000-seed0')

    Returns:
        Base run name without seed (e.g., 'SIMPLe-10000')

    Example:
        >>> get_base_run_name('SIMPLe-10000-seed0')
        'SIMPLe-10000'
        >>> get_base_run_name('AlphaVerus')
        'AlphaVerus'
    """
    base_name = re.sub(r'-seed\d+', '', run_name)
    return base_name


def group_runs_by_seed(run_data: Dict) -> Dict[str, List]:
    """
    Group runs by base name (removing seed suffixes).

    Args:
        run_data: Dictionary mapping run names to run info

    Returns:
        Dictionary mapping base names to lists of run info

    Example:
        >>> run_data = {
        ...     'run-seed0': {'id': '1', 'history': []},
        ...     'run-seed1': {'id': '2', 'history': []},
        ... }
        >>> groups = group_runs_by_seed(run_data)
        >>> len(groups['run'])
        2
    """
    run_groups = {}
    for run_name, run_info in run_data.items():
        base_name = get_base_run_name(run_name)
        if base_name not in run_groups:
            run_groups[base_name] = []
        run_groups[base_name].append(run_info)
    return run_groups


def determine_iteration_for_run(run_name: str) -> int:
    """
    Determine which history array index to use for a given run.

    Args:
        run_name: Run name (e.g., 'AlphaVerus', 'SIMPLe-iter', 'SIMPLe-10000')

    Returns:
        History array index to use (0 for AlphaVerus, 1 for -iter, 5 for others)

    Example:
        >>> determine_iteration_for_run('AlphaVerus')
        0
        >>> determine_iteration_for_run('SIMPLe-iter')
        1
        >>> determine_iteration_for_run('SIMPLe-10000')
        5
    """
    base_name = get_base_run_name(run_name)

    # AlphaVerus uses history[0] (iter1)
    if 'alpha' in base_name.lower():
        return 0

    # -iter ablations use history[1] (iter2)
    if '-iter' in base_name.lower():
        return 1

    # Everything else uses history[5] (iter5)
    return 5


def get_metric_mean_std_values(
    run_name: str,
    dataset: str,
    metric: str,
    run_groups: Dict[str, List]
) -> Tuple[Optional[float], Optional[float], Optional[List[float]]]:
    """
    Get mean, std, and raw values for a metric across all seeds of a run.

    Extracts values at specific history array indices:
    - history[0] (iter1) for AlphaVerus
    - history[1] (iter2) for -iter ablations
    - history[5] (iter5) for all other methods

    Seeds that stopped early (missing the required iteration) are skipped.
    Statistics are computed from available seeds only.

    Args:
        run_name: Run name (with or without seed suffix)
        dataset: Dataset name (e.g., 'DAFNY2VERUS')
        metric: Metric name (e.g., 'pass@1')
        run_groups: Dictionary from group_runs_by_seed()

    Returns:
        Tuple of (mean, std, raw_values), or (None, None, None) if no seeds have data

    Example:
        >>> run_groups = {'run': [
        ...     {'history': [{'DAFNY2VERUS/pass@1': 0.10}, {'DAFNY2VERUS/pass@1': 0.11}, ...]},
        ...     {'history': [{'DAFNY2VERUS/pass@1': 0.12}, {'DAFNY2VERUS/pass@1': 0.13}, ...]},
        ... ]}
        >>> mean, std, values = get_metric_mean_std_values(
        ...     'run-seed0', 'DAFNY2VERUS', 'pass@1', run_groups)
        >>> len(values)
        2
    """
    base_name = get_base_run_name(run_name)

    # Check if this base name exists in run_groups
    if base_name not in run_groups:
        return None, None, None

    # Determine which iteration to use
    iteration = determine_iteration_for_run(run_name)

    metric_key = f"{dataset}/{metric}"
    values = []

    for run_info in run_groups[base_name]:
        value = get_metric_at_iteration(run_info['history'], metric_key, iteration)
        if value is not None:
            # Only include seeds that have the required iteration
            values.append(value)

    if len(values) == 0:
        return None, None, None
    elif len(values) == 1:
        return values[0], 0.0, values
    else:
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        return mean, std, values


def extract_time_series(
    history: List[Dict],
    metric_name: str
) -> Tuple[List, List]:
    """
    Extract time series data (steps and values) for a metric.

    Args:
        history: List of history dictionaries from wandb run
        metric_name: Name of metric to extract

    Returns:
        Tuple of (steps, values) lists

    Example:
        >>> history = [
        ...     {'_step': 0, 'loss': 1.0},
        ...     {'_step': 1, 'loss': 0.8},
        ... ]
        >>> steps, values = extract_time_series(history, 'loss')
        >>> steps
        [0, 1]
        >>> values
        [1.0, 0.8]
    """
    steps = []
    values = []

    for row in history:
        if metric_name in row and row[metric_name] is not None:
            # Handle NaN values stored as strings
            if isinstance(row[metric_name], str) and row[metric_name] == 'NaN':
                continue

            steps.append(row.get('_step', 0))
            values.append(row[metric_name])

    return steps, values
