"""
Formatting utilities for analysis scripts.

Provides functions for formatting values as percentages, with standard deviations,
relative changes, and LaTeX formatting.
"""

import math
from typing import Union, Optional, Tuple


def format_percentage_with_std(
    value_tuple: Union[Tuple, float, None],
    is_best: bool = False
) -> str:
    """
    Format value as percentage with standard deviation as subscript.

    Args:
        value_tuple: Either (mean, std, values), (mean, std), single float, or None
        is_best: If True, bold the main number

    Returns:
        Formatted string like "71.35$_{2.34}$" or "\\textbf{71.35}$_{2.34}$"

    Example:
        >>> format_percentage_with_std((0.7135, 0.0234, []), is_best=False)
        '71.35$_{2.34}$'
        >>> format_percentage_with_std((0.7135, 0.0, []), is_best=True)
        '\\textbf{71.35}'
    """
    # Handle None or NaN
    if value_tuple is None:
        return "—"

    # Unpack mean, std, and values (values might not be present)
    if isinstance(value_tuple, tuple):
        if len(value_tuple) == 3:
            mean, std, _ = value_tuple
        elif len(value_tuple) == 2:
            mean, std = value_tuple
        else:
            mean = value_tuple[0]
            std = 0.0
    else:
        # Backwards compatibility: treat single value as mean with 0 std
        mean, std = value_tuple, 0.0

    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "—"

    # Format mean
    mean_pct = mean * 100
    main_num = f"{mean_pct:.2f}"

    # Bold the main number if best
    if is_best:
        main_num = f"\\textbf{{{main_num}}}"

    # Add std as subscript if non-zero
    if std is not None and std > 0:
        std_pct = std * 100
        formatted = f"{main_num}$_{{{std_pct:.2f}}}$"
    else:
        formatted = main_num

    return formatted


def format_relative_change(
    baseline: Optional[float],
    comparison: Optional[float]
) -> str:
    """
    Format value with relative change arrow.

    Args:
        baseline: Baseline value
        comparison: Comparison value

    Returns:
        Formatted string like "60.00 ($\\uparrow$20.0\\%)" or "40.00 ($\\downarrow$20.0\\%)"

    Example:
        >>> format_relative_change(0.50, 0.60)
        '60.00 ($\\uparrow$20.0\\%)'
        >>> format_relative_change(0.50, 0.40)
        '40.00 ($\\downarrow$20.0\\%)'
    """
    if baseline is None or comparison is None:
        return "—"

    # Format the comparison value
    comparison_pct = comparison * 100
    formatted_value = f"{comparison_pct:.2f}"

    # Compute relative change
    if baseline == 0:
        return formatted_value  # Can't compute relative change

    rel_change = ((comparison - baseline) / baseline) * 100

    if abs(rel_change) < 0.01:  # Essentially zero
        return formatted_value

    # Add arrow based on sign
    if rel_change > 0:
        return f"{formatted_value} ($\\uparrow${abs(rel_change):.1f}\\%)"
    else:
        return f"{formatted_value} ($\\downarrow${abs(rel_change):.1f}\\%)"


def format_value_simple(
    value: Optional[float],
    is_best: bool = False
) -> str:
    """
    Format value as simple percentage.

    Args:
        value: Value to format
        is_best: If True, bold the value

    Returns:
        Formatted string like "71.35" or "\\textbf{71.35}"

    Example:
        >>> format_value_simple(0.7135, is_best=False)
        '71.35'
        >>> format_value_simple(0.7135, is_best=True)
        '\\textbf{71.35}'
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"

    formatted = f"{value * 100:.2f}"

    if is_best:
        return f"\\textbf{{{formatted}}}"

    return formatted
