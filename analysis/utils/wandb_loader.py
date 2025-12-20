"""
W&B data loading utilities for analysis scripts.

Provides functions for connecting to W&B API and loading run data.
"""

import re
import wandb
from typing import Dict, List, Optional


def get_base_run_name(run_name: str) -> str:
    """Extract base run name by removing seed suffix."""
    return re.sub(r'-seed\d+', '', run_name)


def get_seed_number(run_name: str) -> Optional[int]:
    """Extract seed number from run name, or None if no seed suffix."""
    match = re.search(r'-seed(\d+)', run_name)
    return int(match.group(1)) if match else None


def is_run_complete(run_info: Dict) -> bool:
    """
    Check if a run completed successfully using W&B run state.
    A run is complete if its state is 'finished'.
    """
    return run_info.get('state') == 'finished'


def filter_to_first_n_complete_seeds(
    run_data: Dict[str, Dict],
    n_seeds: int = 5
) -> Dict[str, Dict]:
    """
    Filter run data to keep only the first N complete seeds per base run.

    Args:
        run_data: Dictionary mapping run names to run info
        n_seeds: Maximum number of seeds to keep per base run (default: 5)

    Returns:
        Filtered run_data with at most n_seeds per base run
    """
    # Group runs by base name
    groups = {}
    for run_name, run_info in run_data.items():
        base_name = get_base_run_name(run_name)
        if base_name not in groups:
            groups[base_name] = []
        groups[base_name].append((run_name, run_info))

    # Filter each group
    filtered_data = {}
    for base_name, runs in groups.items():
        # Sort by seed number (runs without seed suffix come first)
        runs_sorted = sorted(runs, key=lambda x: get_seed_number(x[0]) or -1)

        # Keep first n_seeds complete runs
        kept = 0
        for run_name, run_info in runs_sorted:
            if kept >= n_seeds:
                break
            if is_run_complete(run_info):
                filtered_data[run_name] = run_info
                kept += 1

        # Log if we couldn't find enough seeds
        if kept < n_seeds and kept > 0:
            print(f"  Warning: {base_name} only has {kept} complete seeds (wanted {n_seeds})")

    return filtered_data


def load_wandb_runs(project_name: str, tags: Optional[List[str]] = None, tags_any: bool = False) -> Dict[str, Dict]:
    """
    Load all runs from a W&B project.

    Args:
        project_name: Name of the W&B project (e.g., "socialiq/simple9")
        tags: Optional list of tags to filter runs by
        tags_any: If True, runs must have ANY of the specified tags (OR logic).
                  If False (default), runs must have ALL specified tags (AND logic).

    Returns:
        Dictionary mapping run names to run info dictionaries.
        Each run info contains: id, name, config, history, tags

    Example:
        >>> run_data = load_wandb_runs("myproject/experiment")
        >>> print(len(run_data))
        10
        >>> print(run_data['run-1']['id'])
        'abc123'
    """
    # Initialize W&B API
    api = wandb.Api()

    # Build filters for tag filtering
    filters = None
    if tags:
        if tags_any:
            filters = {"tags": {"$in": tags}}
        else:
            filters = {"tags": {"$all": tags}}

    # Fetch all runs from the project
    all_runs = api.runs(project_name, filters=filters)

    # Extract run data
    run_data = {}
    for run in all_runs:
        run_info = {
            'id': run.id,
            'name': run.name,
            'config': run.config,
            'history': list(run.scan_history()),
            'tags': run.tags,
            'state': run.state  # 'finished', 'running', 'crashed', 'failed', etc.
        }
        run_data[run.name] = run_info

    return run_data


def load_wandb_runs_with_baselines(
    main_project: str,
    baseline_project: str = "socialiq/simple9",
    baseline_runs: Optional[List[str]] = None,
    n_seeds: int = 5,
    tags: Optional[List[str]] = None,
    tags_any: bool = False
) -> Dict[str, Dict]:
    """
    Load runs from main project plus specified baseline runs from another project.
    Filters to keep only the first n_seeds complete seeds per base run.

    Args:
        main_project: Primary W&B project (e.g., "socialiq/av0")
        baseline_project: Project containing baseline runs (default: "socialiq/simple9")
        baseline_runs: List of base run names to include from baseline project.
                      Defaults to ["AlphaVerus", "REST-EM", "REST-EM-mbpp", "REST-EM-humaneval"]
        n_seeds: Maximum number of seeds to keep per base run (default: 5)
        tags: Optional list of tags to filter main project runs by
        tags_any: If True, runs must have ANY of the specified tags (OR logic).
                  If False (default), runs must have ALL specified tags (AND logic).

    Returns:
        Combined dictionary of run data from both projects, filtered to n_seeds per base run

    Example:
        >>> run_data = load_wandb_runs_with_baselines("socialiq/av0")
        >>> # Contains AV0 runs plus AlphaVerus and REST-EM baselines (5 seeds each)
    """
    if baseline_runs is None:
        baseline_runs = ["AlphaVerus", "REST-EM", "REST-EM-mbpp", "REST-EM-humaneval"]

    # Load main project
    tag_str = f" (tags: {tags}, any={tags_any})" if tags else ""
    print(f"Loading runs from {main_project}{tag_str}...")
    run_data = load_wandb_runs(main_project, tags=tags, tags_any=tags_any)
    print(f"  Found {len(run_data)} runs in {main_project}")

    # Filter main project to first n_seeds complete seeds
    print(f"  Filtering to first {n_seeds} complete seeds per base run...")
    run_data = filter_to_first_n_complete_seeds(run_data, n_seeds)
    print(f"  After filtering: {len(run_data)} runs")

    # Load baseline project
    print(f"Loading baseline runs from {baseline_project}...")
    baseline_data = load_wandb_runs(baseline_project)
    print(f"  Found {len(baseline_data)} runs in {baseline_project}")

    # Filter baseline runs to those we want
    baseline_filtered = {}
    for run_name, run_info in baseline_data.items():
        base_name = get_base_run_name(run_name)
        if base_name in baseline_runs:
            baseline_filtered[run_name] = run_info

    # Filter baseline to first n_seeds complete seeds
    print(f"  Filtering baselines to first {n_seeds} complete seeds per base run...")
    baseline_filtered = filter_to_first_n_complete_seeds(baseline_filtered, n_seeds)

    # Add filtered baseline runs
    for run_name, run_info in baseline_filtered.items():
        run_data[run_name] = run_info

    print(f"  Added {len(baseline_filtered)} baseline runs to dataset")
    print(f"  Total runs: {len(run_data)}")

    return run_data
