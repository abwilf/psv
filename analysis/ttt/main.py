#!/usr/bin/env python3
"""
TTT (Test-Time Training) Analysis Script

Generates:
1. TTT/SIMPLe/{dataset}.png - SIMPLe performance with pass@1,5,10 as different lines (mean ± std across seeds)
2. TTT/Comparison/{dataset}_pass{k}.png - SIMPLe vs REST-EM comparisons (mean ± std across seeds)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import argparse
from typing import Dict, List, Tuple, Optional

# Import utilities
from utils.wandb_loader import load_wandb_runs_with_baselines
from utils.metric_extraction import (
    get_base_run_name,
    group_runs_by_seed,
    extract_time_series
)

# Parse arguments
parser = argparse.ArgumentParser(description='Generate TTT analysis plots')
parser.add_argument('--load_from_cache', action='store_true',
                    help='Load run data from cache if available')
args = parser.parse_args()

# Create output directories
os.makedirs('figs', exist_ok=True)
os.makedirs('analyses', exist_ok=True)

CACHE_PATH = 'analysis/ttt/run_cache.pk'

# ============================================================================
# LOAD DATA FROM WANDB (or cache)
# ============================================================================
if args.load_from_cache and os.path.exists(CACHE_PATH):
    print("=== Loading from cache ===")
    with open(CACHE_PATH, 'rb') as f:
        run_data = pickle.load(f)
    print(f"Loaded {len(run_data)} runs from cache")
else:
    print("=== Connecting to W&B ===")
    print("\nFetching all runs from socialiq/av0 (with baselines from simple9)...")
    run_data = load_wandb_runs_with_baselines("socialiq/av0")

    # Save to cache
    print(f"\nSaving {len(run_data)} runs to cache: {CACHE_PATH}")
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(run_data, f)

print(f"\n=== Found {len(run_data)} total runs ===\n")
print("Available run names:")
print("-" * 80)
for idx, run_name in enumerate(sorted(run_data.keys()), 1):
    tags_str = ', '.join(run_data[run_name]['tags']) if run_data[run_name]['tags'] else 'no tags'
    print(f"{idx:3d}. {run_name:40s} | Tags: {tags_str}")
print("-" * 80)

# Group runs by seed
print("\n=== Grouping runs by seed ===")
run_groups = group_runs_by_seed(run_data)

print(f"Grouped {len(run_data)} runs into {len(run_groups)} base configurations")
for base_name, runs in sorted(run_groups.items()):
    print(f"  {base_name}: {len(runs)} seed(s)")


def get_time_series_across_seeds(base_run_name: str, dataset: str, metric: str) -> Tuple[List[int], List[float], List[float]]:
    """
    Get time series data aggregated across all seeds for a run.

    Args:
        base_run_name: Base run name (without seed suffix)
        dataset: Dataset name (e.g., 'DAFNY2VERUS')
        metric: Metric name (e.g., 'pass@1')

    Returns:
        Tuple of (steps, mean_values, std_values)
    """
    if base_run_name not in run_groups:
        return [], [], []

    metric_key = f"{dataset}/{metric}"

    # Extract time series for each seed
    all_series = []
    for run_info in run_groups[base_run_name]:
        steps, values = extract_time_series(run_info['history'], metric_key)
        if steps and values:
            all_series.append((steps, values))

    if not all_series:
        return [], [], []

    # Find common steps and interpolate
    all_steps = sorted(set(step for steps, _ in all_series for step in steps))

    # For each step, collect values from all seeds
    mean_values = []
    std_values = []

    for step in all_steps:
        step_values = []
        for steps, values in all_series:
            if step in steps:
                idx = steps.index(step)
                step_values.append(values[idx])

        if step_values:
            mean_values.append(np.mean(step_values))
            std_values.append(np.std(step_values, ddof=1) if len(step_values) > 1 else 0.0)
        else:
            mean_values.append(np.nan)
            std_values.append(np.nan)

    return all_steps, mean_values, std_values


def plot_simple_lines(dataset_name: str, base_run_name: str, metrics: List[str], output_file: str):
    """
    Plot a single run with multiple pass@k metrics as different lines (mean ± std across seeds).

    Args:
        dataset_name: Name of dataset (e.g., 'DAFNY2VERUS', 'MBPP', 'HUMANEVAL')
        base_run_name: Base run name (without seed suffix)
        metrics: List of pass@k values (e.g., ['pass@1', 'pass@5', 'pass@10'])
        output_file: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    markers = ['o', 's', '^']

    if base_run_name not in run_groups:
        print(f"  Warning: '{base_run_name}' not found in run groups")
        return

    plotted_any = False
    for metric_idx, metric_suffix in enumerate(metrics):
        steps, mean_values, std_values = get_time_series_across_seeds(base_run_name, dataset_name, metric_suffix)

        if steps and mean_values:
            # Plot mean line
            ax.plot(steps, mean_values,
                   label=metric_suffix,
                   color=colors[metric_idx % len(colors)],
                   linestyle='-',
                   marker=markers[metric_idx % len(markers)],
                   linewidth=2.5,
                   markersize=8,
                   markevery=1)

            # Plot shaded error region (mean ± std)
            ax.fill_between(steps,
                          np.array(mean_values) - np.array(std_values),
                          np.array(mean_values) + np.array(std_values),
                          color=colors[metric_idx % len(colors)],
                          alpha=0.2)

            plotted_any = True
            n_seeds = len(run_groups[base_run_name])
            print(f"  ✓ Plotted {metric_suffix} for {base_run_name} ({n_seeds} seeds)")
        else:
            print(f"  ✗ No data for {base_run_name} on {dataset_name}/{metric_suffix}")

    if plotted_any:
        ax.set_xlabel('Iteration', fontsize=32)
        ax.set_ylabel('Pass Rate', fontsize=32)
        # Map dataset keys to display names
        display_names = {'DAFNY2VERUS': 'Dafny2Verus', 'MBPP': 'MBPP', 'HUMANEVAL': 'HumanEval'}
        display_name = display_names.get(dataset_name, dataset_name)
        ax.set_title(f"{display_name}", fontsize=32)
        ax.legend(fontsize=18, loc='best', frameon=True, shadow=True)
        ax.tick_params(axis='both', labelsize=24)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}\n")
    else:
        plt.close()


def plot_comparison(dataset_name: str, metric: str, runs_config: List[Dict], output_file: str):
    """
    Plot comparison between different runs for a specific metric (mean ± std across seeds).

    Args:
        dataset_name: Name of dataset (e.g., 'DAFNY2VERUS', 'MBPP', 'HUMANEVAL')
        metric: pass@k metric (e.g., 'pass@1', 'pass@5', 'pass@10')
        runs_config: List of dicts with 'name', 'label', 'color', 'linestyle', 'marker'
                    Optional 'horizontal': True for runs that should be plotted as horizontal lines
        output_file: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    plotted_any = False
    all_steps = []  # Track all steps for horizontal line plotting
    horizontal_runs = []  # Store horizontal runs to plot after we know x-range

    for run_config in runs_config:
        base_run_name = run_config['name']

        if base_run_name not in run_groups:
            print(f"  Warning: '{base_run_name}' not found in run groups")
            continue

        steps, mean_values, std_values = get_time_series_across_seeds(base_run_name, dataset_name, metric)

        if steps and mean_values:
            all_steps.extend(steps)

            # Check if this should be a horizontal baseline
            if run_config.get('horizontal', False):
                # Store for later plotting once we know full x-range
                horizontal_runs.append({
                    'config': run_config,
                    'value': mean_values[0],  # Use the single value
                    'std': std_values[0] if std_values else 0
                })
                continue

            # Plot mean line
            ax.plot(steps, mean_values,
                   label=run_config['label'],
                   color=run_config['color'],
                   linestyle=run_config['linestyle'],
                   marker=run_config['marker'],
                   linewidth=4,
                   markersize=14,
                   markevery=1)

            # Plot shaded error region (mean ± std)
            ax.fill_between(steps,
                          np.array(mean_values) - np.array(std_values),
                          np.array(mean_values) + np.array(std_values),
                          color=run_config['color'],
                          alpha=0.2)

            plotted_any = True
            n_seeds = len(run_groups[base_run_name])
            print(f"  ✓ Plotted {base_run_name} for {dataset_name}/{metric} ({n_seeds} seeds)")
        else:
            print(f"  ✗ No data for {base_run_name} on {dataset_name}/{metric}")

    # Plot horizontal baselines across the x-range of other runs
    if all_steps and horizontal_runs:
        x_min, x_max = min(all_steps), max(all_steps)
        for hr in horizontal_runs:
            # Use hlines instead of axhline to limit to x_max
            ax.hlines(y=hr['value'], xmin=x_min, xmax=x_max,
                     label=hr['config']['label'],
                     color=hr['config']['color'],
                     linestyle=hr['config']['linestyle'],
                     linewidth=4)
            # Add shaded error region for horizontal line (limited to x-range)
            if hr['std'] > 0:
                ax.fill_between([x_min, x_max],
                              [hr['value'] - hr['std']] * 2,
                              [hr['value'] + hr['std']] * 2,
                              color=hr['config']['color'],
                              alpha=0.2)
            plotted_any = True
            print(f"  ✓ Plotted {hr['config']['name']} as horizontal baseline for {dataset_name}/{metric}")

    if plotted_any:
        ax.set_xlabel('Iteration', fontsize=40)
        ylabel = metric.replace('@', ' @ ')
        ylabel = ylabel[0].upper() + ylabel[1:]  # Capitalize first letter
        ax.set_ylabel(ylabel, fontsize=40)
        # Map dataset keys to display names
        display_names = {'DAFNY2VERUS': 'Dafny2Verus', 'MBPP': 'MBPP', 'HUMANEVAL': 'HumanEval'}
        display_name = display_names.get(dataset_name, dataset_name)
        ax.set_title(f"{display_name}", fontsize=40, fontweight='bold')
        ax.legend(fontsize=20, loc='best', frameon=True, shadow=True)
        ax.tick_params(axis='both', labelsize=28)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}\n")
    else:
        plt.close()


# ============================================================================
# PART 1: TTT/AV0 plots (pass@1,5,10 as different lines)
# ============================================================================
print("\n=== Creating TTT/AV0 plots ===")

datasets = [
    ('DAFNY2VERUS', 'AV0', 'Dafny2Verus'),
    ('MBPP', 'AV0-mbpp', 'MBPP'),
    ('HUMANEVAL', 'AV0-humaneval', 'HumanEval')
]

for dataset_key, run_name, display_name in datasets:
    print(f"\nPlotting {display_name}...")
    plot_simple_lines(
        dataset_key,
        run_name,
        ['pass@1', 'pass@5', 'pass@10'],
        f'figs/ttt_av0_{display_name.lower()}.png'
    )

# ============================================================================
# PART 2: TTT/Comparison plots (AV0 vs REST-EM for each dataset/pass@ combo)
# ============================================================================
print("\n=== Creating TTT/Comparison plots ===")

comparison_runs = [
    {
        'name': 'AV0',
        'label': 'PSV',
        'color': '#2E86AB',  # Blue
        'linestyle': '-',
        'marker': 'o'
    },
    {
        'name': 'REST-EM',
        'label': 'RFT',
        'color': '#FF8C42',  # Orange
        'linestyle': '-',
        'marker': 's'
    },
    # {
    #     'name': 'AV0-diff',
    #     'label': 'AV0-diff',
    #     'color': '#E63946',  # Red
    #     'linestyle': '-',
    #     'marker': '^'
    # },
    # {
    #     'name': 'AV0-sampling',
    #     'label': 'AV0-sampling',
    #     'color': '#2A9D8F',  # Teal
    #     'linestyle': '-',
    #     'marker': 'D'
    # },
    {
        'name': 'AlphaVerus',
        'label': 'AlphaVerus',
        'color': '#6B7280',  # Gray
        'linestyle': ':',
        'marker': None,
        'horizontal': True  # Special flag for horizontal baseline
    }
]

comparison_runs_mbpp = [
    {
        'name': 'AV0-mbpp',
        'label': 'PSV',
        'color': '#2E86AB',  # Blue
        'linestyle': '-',
        'marker': 'o'
    },
    {
        'name': 'REST-EM-mbpp',
        'label': 'RFT',
        'color': '#FF8C42',  # Orange
        'linestyle': '-',
        'marker': 's'
    },
    {
        'name': 'AlphaVerus',
        'label': 'AlphaVerus',
        'color': '#6B7280',  # Gray
        'linestyle': ':',
        'marker': None,
        'horizontal': True  # Special flag for horizontal baseline
    }
]

comparison_runs_humaneval = [
    {
        'name': 'AV0-humaneval',
        'label': 'AV0',
        'color': '#2E86AB',
        'linestyle': '-',
        'marker': 'o'
    },
    {
        'name': 'REST-EM-humaneval',
        'label': 'REST-EM',
        'color': '#FF8C42',
        'linestyle': '-',
        'marker': 's'
    }
]

# Dafny2Verus comparisons
for metric in ['pass@1', 'pass@5', 'pass@10']:
    metric_name = metric.replace('@', '')
    print(f"\nPlotting Dafny2Verus {metric}...")
    plot_comparison(
        'DAFNY2VERUS',
        metric,
        comparison_runs,
        f'figs/ttt_comparison_dafny2verus_{metric_name}.png'
    )

# MBPP comparisons
for metric in ['pass@1', 'pass@5', 'pass@10']:
    metric_name = metric.replace('@', '')
    print(f"\nPlotting MBPP {metric}...")
    plot_comparison(
        'MBPP',
        metric,
        comparison_runs_mbpp,
        f'figs/ttt_comparison_mbpp_{metric_name}.png'
    )

# HumanEval comparisons
for metric in ['pass@1', 'pass@5', 'pass@10']:
    metric_name = metric.replace('@', '')
    print(f"\nPlotting HumanEval {metric}...")
    plot_comparison(
        'HUMANEVAL',
        metric,
        comparison_runs_humaneval,
        f'figs/ttt_comparison_humaneval_{metric_name}.png'
    )

print("\n" + "=" * 80)
print("=== TTT Analysis Complete! ===")
print("=" * 80)
print("\nGenerated plots:")
print("\nAV0 (pass@1,5,10 as lines):")
print("  - figs/ttt_av0_dafny2verus.png")
print("  - figs/ttt_av0_mbpp.png")
print("  - figs/ttt_av0_humaneval.png")
print("\nComparison (AV0 vs REST-EM):")
print("  - figs/ttt_comparison_dafny2verus_pass{1,5,10}.png")
print("  - figs/ttt_comparison_mbpp_pass{1,5,10}.png")
print("  - figs/ttt_comparison_humaneval_pass{1,5,10}.png")
print()


# ============================================================================
# STATISTICS FOR RESULTS PARAGRAPH
# ============================================================================
print("\n" + "=" * 80)
print("=== STATISTICS FOR RESULTS PARAGRAPH ===")
print("=" * 80)


def get_final_value_across_seeds(base_run_name: str, dataset: str, metric: str) -> Optional[float]:
    """Get the final (last iteration) mean value across all seeds."""
    steps, mean_values, std_values = get_time_series_across_seeds(base_run_name, dataset, metric)
    if mean_values:
        return mean_values[-1]
    return None


# TTT run configurations for statistics
ttt_stats_configs = [
    ('DAFNY2VERUS', 'AV0', 'REST-EM', 'D2V'),
    ('MBPP', 'AV0-mbpp', 'REST-EM-mbpp', 'MBPP'),
    ('HUMANEVAL', 'AV0-humaneval', 'REST-EM-humaneval', 'HE')
]

metrics = ['pass@1', 'pass@5', 'pass@10']

print("\n--- TTT Statistics (Test-Time Training) ---")
print("(Models trained on each specific dataset)\n")

# Per-dataset statistics
all_av0_values = []
all_rest_values = []

for dataset_key, av0_run, rest_run, short_name in ttt_stats_configs:
    print(f"\n{dataset_key}:")
    for metric in metrics:
        av0_val = get_final_value_across_seeds(av0_run, dataset_key, metric)
        rest_val = get_final_value_across_seeds(rest_run, dataset_key, metric)

        if av0_val is not None:
            all_av0_values.append(av0_val)
        if rest_val is not None:
            all_rest_values.append(rest_val)

        av0_str = f"{av0_val*100:.2f}%" if av0_val else "N/A"
        rest_str = f"{rest_val*100:.2f}%" if rest_val else "N/A"

        if av0_val and rest_val and rest_val > 0:
            ratio = av0_val / rest_val
            diff_pp = (av0_val - rest_val) * 100
            print(f"  {metric}: AV0={av0_str}  REST-EM={rest_str}  (AV0/REST={ratio:.2f}x, +{diff_pp:.2f}pp)")
        else:
            print(f"  {metric}: AV0={av0_str}  REST-EM={rest_str}")

# Summary across all datasets and metrics
print("\n--- TTT Averages (all benchmarks, all metrics) ---")
if all_av0_values and all_rest_values:
    av0_avg = sum(all_av0_values) / len(all_av0_values)
    rest_avg = sum(all_rest_values) / len(all_rest_values)

    print(f"Average: AV0={av0_avg*100:.2f}%  REST-EM={rest_avg*100:.2f}%")

    if rest_avg > 0:
        mean_ratio = av0_avg / rest_avg
        mean_diff = (av0_avg - rest_avg) * 100
        print(f"Mean improvement: vs REST-EM={mean_ratio:.2f}x (+{mean_diff:.2f}pp)")

# Max improvements
print("\n--- TTT Max Improvements ---")
max_ratio = 0
max_diff = 0
for dataset_key, av0_run, rest_run, short_name in ttt_stats_configs:
    for metric in metrics:
        av0_val = get_final_value_across_seeds(av0_run, dataset_key, metric)
        rest_val = get_final_value_across_seeds(rest_run, dataset_key, metric)

        if av0_val and rest_val and rest_val > 0:
            ratio = av0_val / rest_val
            diff = av0_val - rest_val
            if ratio > max_ratio:
                max_ratio = ratio
            if diff > max_diff:
                max_diff = diff

print(f"Max improvement: vs REST-EM={max_ratio:.2f}x (+{max_diff*100:.2f}pp)")

print("\n" + "=" * 80)


