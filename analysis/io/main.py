#!/usr/bin/env python3
"""
IO (Input/Output Transfer) Analysis Script

Compares different input/output difficulty transfer strategies:
- a2e: All difficulties to easy
- a2h: All difficulties to hard
- a2u: All difficulties to unsolved
- e2h: Easy to hard
- h2h: Hard to hard
"""

import wandb
import pandas as pd
import os
import sys
import pickle
import argparse
from typing import Dict, List, Optional, Tuple

# Add path to utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wandb_loader import load_wandb_runs, filter_to_first_n_complete_seeds
from utils.metric_extraction import group_runs_by_seed, get_metric_mean_std_values
from utils.statistics import find_best_with_significance
from utils.formatting import format_percentage_with_std

# Parse arguments
parser = argparse.ArgumentParser(description='Generate IO transfer analysis table')
parser.add_argument('--load_from_cache', action='store_true',
                    help='Load run data from cache if available')
args = parser.parse_args()

# Create output directories
os.makedirs('tables', exist_ok=True)
os.makedirs('analyses', exist_ok=True)

CACHE_PATH = 'analysis/io/run_cache.pk'

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
    print("\nFetching all runs from socialiq/av0...")
    run_data = load_wandb_runs("socialiq/av0")
    print(f"  Found {len(run_data)} runs")

    # Filter to first 5 complete seeds per base run
    print("  Filtering to first 5 complete seeds per base run...")
    run_data = filter_to_first_n_complete_seeds(run_data, n_seeds=5)

    # Save to cache
    print(f"\nSaving {len(run_data)} runs to cache: {CACHE_PATH}")
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(run_data, f)

print(f"\n=== Loaded data for {len(run_data)} runs ===\n")

print("=== Grouping runs by seed ===")
run_groups = group_runs_by_seed(run_data)
print(f"Grouped into {len(run_groups)} base experiment names")


def get_metric(run_name: str, dataset: str, metric: str) -> Tuple[Optional[float], Optional[float], Optional[List[float]]]:
    """
    Get mean, std, and raw values for a metric across all seeds.
    Returns: (mean, std, raw_values) or (None, None, None) if data missing.
    """
    mean, std, raw_values = get_metric_mean_std_values(
        run_name=run_name,
        dataset=dataset,
        metric=metric,
        run_groups=run_groups
    )

    if mean is not None:
        print(f"  ✓ {run_name}: {dataset}/{metric} = {mean:.4f} ± {std:.4f} (n={len(raw_values)})")
    else:
        print(f"  ✗ No data for {run_name}: {dataset}/{metric} (missing seeds or iteration)")

    return mean, std, raw_values


# Note: format_value and find_best_in_column removed - now using utility functions:
# - format_percentage_with_std() from utils.formatting
# - find_best_with_significance() from utils.statistics


# ============================================================================
# IDENTIFY IO TRANSFER RUNS
# ============================================================================
print("\n" + "=" * 80)
print("=== Identifying IO Transfer Runs ===")
print("=" * 80)

# IO transfer runs configuration
# Format: (run_name, display_label)
# Note: Base names only - utilities will automatically find all seeds (0-4)
io_runs = [
    ('AV0_ai2h', 'A2H'),
    ('AV0_e2h', 'E2H'),
    ('AV0_ai2e', 'A2E'),
    ('AV0', 'A2U'),
    ('AV0-diff', '-diff'),
    ('AV0-sampling', '-samp'),
]

# ============================================================================
# TABLE: IO TRANSFER COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("=== Creating IO Transfer Comparison Table ===")
print("=" * 80)

datasets = ['DAFNY2VERUS', 'MBPP', 'HUMANEVAL']
metrics = ['pass@1', 'pass@5', 'pass@10']

# Collect data for IO table
io_data = []

for run_name, label in io_runs:
    row_data = {'Method': label}

    print(f"\nProcessing {run_name} (label: {label})...")

    for dataset in datasets:
        for metric in metrics:
            col_key = f"{dataset}/{metric}"
            value = get_metric(run_name, dataset, metric)
            row_data[col_key] = value

    io_data.append(row_data)

# Generate LaTeX for IO Transfer
print("\n=== Generating IO Transfer LaTeX Table ===")

# Find statistically significant best value in each column
print("\n=== Finding statistically significant best values ===")
best_indices = {}
for dataset in datasets:
    for metric in metrics:
        col_key = f"{dataset}/{metric}"
        # Extract (mean, std, raw_values) tuples for this column
        column_values = [row[col_key] for row in io_data]
        # Find the index of the statistically best value
        best_idx = find_best_with_significance(column_values, alpha=0.05)
        best_indices[col_key] = best_idx
        if best_idx >= 0:
            print(f"  ✓ {col_key}: Best = {io_data[best_idx]['Method']} (statistically significant)")
        else:
            print(f"  - {col_key}: No statistically significant best")

latex_io = "\\begin{table*}[t]\n"
latex_io += "\\centering\n"
latex_io += "\\caption{Input/Output Prompting Settings. Mean pass@k with standard errors across 5 seeds. A=All, E=Easy, H=Hard, U=Uniform.}\n"
latex_io += "\\label{tab:io_results}\n"
latex_io += "\\setlength{\\tabcolsep}{3pt}\n"
latex_io += "\\begin{tabular}{l|ccc|ccc|ccc|c}\n"
latex_io += "\\hline\n"
latex_io += "& \\multicolumn{3}{c|}{\\textbf{Dafny2Verus}} & \\multicolumn{3}{c|}{\\textbf{MBPP}} & \\multicolumn{3}{c|}{\\textbf{HumanEval}} & \\\\\n"
latex_io += "& pass@1 & pass@5 & pass@10 & pass@1 & pass@5 & pass@10 & pass@1 & pass@5 & pass@10 & \\textbf{Avg} \\\\\n"
latex_io += "\\hline\n"

for row_idx, row in enumerate(io_data):
    latex_io += f"{row['Method']}"

    # Collect mean values for average calculation
    mean_values = []
    for dataset in datasets:
        for metric in metrics:
            col_key = f"{dataset}/{metric}"
            value_tuple = row[col_key]  # (mean, std, raw_values)
            if value_tuple[0] is not None:  # Check if mean exists
                mean_values.append(value_tuple[0])

    # Calculate average of means
    if mean_values:
        avg_mean = sum(mean_values) / len(mean_values)
        avg_formatted = f"{avg_mean * 100:.2f}"
    else:
        avg_formatted = "—"

    # Add metric columns
    for dataset in datasets:
        for metric in metrics:
            col_key = f"{dataset}/{metric}"
            value_tuple = row[col_key]  # (mean, std, raw_values)
            is_best = (best_indices[col_key] == row_idx)

            # Format with standard error and bolding if best
            formatted_value = format_percentage_with_std(value_tuple, is_best)
            latex_io += f" & {formatted_value}"

    # Add average column
    latex_io += f" & {avg_formatted}"
    latex_io += " \\\\\n"

latex_io += "\\hline\n"
latex_io += "\\end{tabular}\n"
latex_io += "\\end{table*}\n"

# Save IO table
with open('tables/io_transfer.tex', 'w') as f:
    f.write(latex_io)

print("✓ Saved: tables/io_transfer.tex")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("=== IO Transfer Analysis Complete! ===")
print("=" * 80)
print("\nGenerated table:")
print("  1. tables/io_transfer.tex")
print()


# ============================================================================
# STATISTICS FOR RESULTS PARAGRAPH
# ============================================================================
# ============================================================================
# PROPOSAL AND TRAINING STATISTICS (AV0 vs AV0-sampling)
# ============================================================================
print("\n" + "=" * 80)
print("=== PROPOSAL AND TRAINING STATISTICS ===")
print("=" * 80)

def get_proposal_stats(base_run_name: str):
    """Get proposal and training statistics across all seeds for a run."""
    stats = {
        'n_generated': [],
        'n_unique': [],
        'n_valid': [],
        'uniqueness_rate': [],
        'valid_rate': [],
        'len_trn_data': [],
    }

    for run_name, run_info in run_data.items():
        # Check if this run belongs to the base run
        from utils.wandb_loader import get_base_run_name
        if get_base_run_name(run_name) != base_run_name:
            continue

        # Get the last iteration's data
        history = run_info.get('history', [])
        if not history:
            continue

        # Get the last entry (final iteration)
        last_entry = history[-1]

        for key in stats.keys():
            metric_key = f'proposal/{key}' if key != 'len_trn_data' else key
            if metric_key in last_entry and last_entry[metric_key] is not None:
                stats[key].append(last_entry[metric_key])

    # Compute means
    result = {}
    for key, values in stats.items():
        if values:
            result[key] = {
                'mean': sum(values) / len(values),
                'values': values,
                'n': len(values)
            }
        else:
            result[key] = None

    return result

# Get stats for AV0 and AV0-sampling
av0_stats = get_proposal_stats('AV0')
sampling_stats = get_proposal_stats('AV0-sampling')

print("\n--- AV0 (Our Method) ---")
if av0_stats:
    for key, data in av0_stats.items():
        if data:
            print(f"  {key}: {data['mean']:.2f} (n={data['n']})")

print("\n--- AV0-sampling (Fixed Prompt) ---")
if sampling_stats:
    for key, data in sampling_stats.items():
        if data:
            print(f"  {key}: {data['mean']:.2f} (n={data['n']})")

# Compare the two
print("\n--- Comparison (AV0 vs AV0-sampling) ---")
if av0_stats and sampling_stats:
    for key in ['n_unique', 'n_valid', 'len_trn_data']:
        av0_data = av0_stats.get(key)
        samp_data = sampling_stats.get(key)
        if av0_data and samp_data:
            ratio = av0_data['mean'] / samp_data['mean'] if samp_data['mean'] > 0 else float('inf')
            diff = av0_data['mean'] - samp_data['mean']
            print(f"  {key}: AV0={av0_data['mean']:.1f}, Sampling={samp_data['mean']:.1f} (AV0 is {ratio:.2f}x, +{diff:.1f})")

print("\n" + "=" * 80)
print("=== STATISTICS FOR RESULTS PARAGRAPH ===")
print("=" * 80)


def get_mean_from_tuple(value_tuple):
    """Extract mean from (mean, std, raw_values) tuple."""
    if value_tuple is None:
        return None
    if isinstance(value_tuple, tuple) and value_tuple[0] is not None:
        return value_tuple[0]
    return None


# Baseline is All -> Uniform (AV0)
baseline_label = 'All $\\rightarrow$ Uniform'
baseline_idx = next((i for i, row in enumerate(io_data) if row['Method'] == baseline_label), None)

print("\n--- IO Transfer Strategy Comparison ---")
print("(Comparing different input/output difficulty transfer strategies)\n")

# Print per-strategy results
for row in io_data:
    method = row['Method']
    print(f"\n{method}:")

    # Compute average across all metrics
    means = []
    for dataset in datasets:
        for metric in metrics:
            col_key = f"{dataset}/{metric}"
            mean_val = get_mean_from_tuple(row[col_key])
            if mean_val is not None:
                means.append(mean_val)

    if means:
        avg = sum(means) / len(means)
        print(f"  Average (all metrics): {avg*100:.2f}%")

    # Print pass@1 for each dataset
    for dataset in datasets:
        col_key = f"{dataset}/pass@1"
        mean_val = get_mean_from_tuple(row[col_key])
        if mean_val is not None:
            print(f"  {dataset} pass@1: {mean_val*100:.2f}%")

# Compare strategies vs baseline
if baseline_idx is not None:
    print("\n--- Improvements vs Baseline (All -> Uniform) ---")

    baseline_row = io_data[baseline_idx]
    baseline_means = []
    for dataset in datasets:
        for metric in metrics:
            col_key = f"{dataset}/{metric}"
            mean_val = get_mean_from_tuple(baseline_row[col_key])
            if mean_val is not None:
                baseline_means.append(mean_val)

    baseline_avg = sum(baseline_means) / len(baseline_means) if baseline_means else 0

    for row in io_data:
        if row['Method'] == baseline_label:
            continue

        method = row['Method']
        means = []
        for dataset in datasets:
            for metric in metrics:
                col_key = f"{dataset}/{metric}"
                mean_val = get_mean_from_tuple(row[col_key])
                if mean_val is not None:
                    means.append(mean_val)

        if means:
            avg = sum(means) / len(means)
            if baseline_avg > 0:
                ratio = avg / baseline_avg
                diff_pp = (avg - baseline_avg) * 100
                print(f"  {method}: {avg*100:.2f}% ({ratio:.2f}x vs baseline, {diff_pp:+.2f}pp)")

# Find best strategy
print("\n--- Best Strategy per Dataset (pass@1) ---")
for dataset in datasets:
    col_key = f"{dataset}/pass@1"
    best_val = 0
    best_method = None
    for row in io_data:
        mean_val = get_mean_from_tuple(row[col_key])
        if mean_val is not None and mean_val > best_val:
            best_val = mean_val
            best_method = row['Method']
    if best_method:
        print(f"  {dataset}: {best_method} ({best_val*100:.2f}%)")

print("\n" + "=" * 80)


