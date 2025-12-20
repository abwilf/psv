"""
Analysis utilities for AV2 project.

This package provides reusable utilities for wandb data loading,
metric extraction, statistical analysis, formatting, and LaTeX generation.
"""

# Import metric_extraction (implemented)
from .metric_extraction import (
    get_final_metric_value,
    get_base_run_name,
    group_runs_by_seed,
    get_metric_mean_std_values,
    extract_time_series
)

__all__ = [
    # metric_extraction
    'get_final_metric_value',
    'get_base_run_name',
    'group_runs_by_seed',
    'get_metric_mean_std_values',
    'extract_time_series',
]

# Import other modules as they are implemented
try:
    from .statistics import (
        find_best_in_column,
        is_significantly_better,
        find_best_with_significance,
        compute_relative_change,
        compute_absolute_change
    )
    __all__.extend([
        'find_best_in_column',
        'is_significantly_better',
        'find_best_with_significance',
        'compute_relative_change',
        'compute_absolute_change',
    ])
except (ImportError, AttributeError):
    pass

try:
    from .formatting import (
        format_percentage_with_std,
        format_relative_change,
        format_value_simple
    )
    __all__.extend([
        'format_percentage_with_std',
        'format_relative_change',
        'format_value_simple',
    ])
except (ImportError, AttributeError):
    pass

try:
    from .latex_generation import (
        create_latex_table_header,
        create_latex_table_footer,
        format_multicolumn_header,
        save_latex
    )
    __all__.extend([
        'create_latex_table_header',
        'create_latex_table_footer',
        'format_multicolumn_header',
        'save_latex',
    ])
except (ImportError, AttributeError):
    pass

try:
    from .wandb_loader import (
        load_wandb_runs
    )
    __all__.extend([
        'load_wandb_runs',
    ])
except (ImportError, AttributeError):
    pass
