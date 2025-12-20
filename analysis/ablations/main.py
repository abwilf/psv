#!/usr/bin/env python3
"""
Ablations Analysis Script

Compares AlphaVerus-Zero with ablations:
- AlphaVerus-Zero (main method)
- -difficulty (difficulty-based prompting ablation)
- -sampling (sampling strategy ablation)
- -solution_verification (solution verification ablation)
- -solution&spec_verification (both verification types ablation)

Uses same table format as IO analysis (pass@1, pass@5, pass@10 for each dataset + Avg).
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
from utils.wandb_loader import load_wandb_runs, filter_to_first_n_complete_seeds, get_base_run_name
from utils.metric_extraction import group_runs_by_seed, get_metric_mean_std_values
from utils.statistics import find_best_with_significance, compute_pvalue
from utils.formatting import format_percentage_with_std

# Parse arguments
parser = argparse.ArgumentParser(description='Generate ablations analysis table')
parser.add_argument('--load_from_cache', action='store_true',
                    help='Load run data from cache if available')
args = parser.parse_args()

# Create output directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(SCRIPT_DIR, exist_ok=True)

CACHE_PATH = os.path.join(SCRIPT_DIR, 'run_cache.pk')

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


# ============================================================================
# IDENTIFY ABLATION RUNS
# ============================================================================
print("\n" + "=" * 80)
print("=== Identifying Ablation Runs ===")
print("=" * 80)

# Ablation runs configuration
# Format: (run_name, display_label)
ablation_runs = [
    ('AV0', 'AlphaVerus-Zero'),
    ('AV0-diff', '-difficulty'),
    ('AV0-sampling2', '-sampling'),
    ('AV0-solutionver', '-solution'),
    ('AV0-ver', '-solution\\&spec'),
]

# ============================================================================
# TABLE: ABLATIONS COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("=== Creating Ablations Comparison Table ===")
print("=" * 80)

datasets = ['DAFNY2VERUS', 'MBPP', 'HUMANEVAL']
metrics = ['pass@1', 'pass@5', 'pass@10']

# Collect data for ablations table
ablation_data = []

for run_name, label in ablation_runs:
    row_data = {'Method': label}

    print(f"\nProcessing {run_name} (label: {label})...")

    for dataset in datasets:
        for metric in metrics:
            col_key = f"{dataset}/{metric}"
            value = get_metric(run_name, dataset, metric)
            row_data[col_key] = value

    ablation_data.append(row_data)

# Generate LaTeX for Ablations
print("\n=== Generating Ablations LaTeX Table ===")

# Find statistically significant best value in each column
print("\n=== Finding statistically significant best values ===")
best_indices = {}
for dataset in datasets:
    for metric in metrics:
        col_key = f"{dataset}/{metric}"
        # Extract (mean, std, raw_values) tuples for this column
        column_values = [row[col_key] for row in ablation_data]
        # Find the index of the statistically best value
        best_idx = find_best_with_significance(column_values, alpha=0.05)
        best_indices[col_key] = best_idx
        if best_idx >= 0:
            print(f"  ✓ {col_key}: Best = {ablation_data[best_idx]['Method']} (statistically significant)")
        else:
            print(f"  - {col_key}: No statistically significant best")

latex_ablations = "\\begin{table*}[t]\n"
latex_ablations += "\\centering\n"
latex_ablations += "\\caption{Ablation Study. Mean pass@k with standard errors across 5 seeds. Best results (statistically significant via paired t-test, $p < 0.05$) are \\textbf{bolded}.}\n"
latex_ablations += "\\label{tab:ablation_results}\n"
latex_ablations += "\\setlength{\\tabcolsep}{3pt}\n"
latex_ablations += "\\begin{tabular}{l|ccc|ccc|ccc|c}\n"
latex_ablations += "\\hline\n"
latex_ablations += "& \\multicolumn{3}{c|}{\\textbf{Dafny2Verus}} & \\multicolumn{3}{c|}{\\textbf{MBPP}} & \\multicolumn{3}{c|}{\\textbf{HumanEval}} & \\\\\n"
latex_ablations += "& pass@1 & pass@5 & pass@10 & pass@1 & pass@5 & pass@10 & pass@1 & pass@5 & pass@10 & \\textbf{Avg} \\\\\n"
latex_ablations += "\\hline\n"

for row_idx, row in enumerate(ablation_data):
    latex_ablations += f"{row['Method']}"

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
            latex_ablations += f" & {formatted_value}"

    # Add average column
    latex_ablations += f" & {avg_formatted}"
    latex_ablations += " \\\\\n"

latex_ablations += "\\hline\n"
latex_ablations += "\\end{tabular}\n"
latex_ablations += "\\end{table*}\n"

# Save ablations table
output_path = os.path.join(SCRIPT_DIR, 'ablations.tex')
with open(output_path, 'w') as f:
    f.write(latex_ablations)

print(f"✓ Saved: {output_path}")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("=== Ablations Analysis Complete! ===")
print("=" * 80)
print("\nGenerated table:")
print(f"  1. {output_path}")
print()


# ============================================================================
# STATISTICS FOR RESULTS PARAGRAPH
# ============================================================================
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


# Baseline is AlphaVerus-Zero (AV0)
baseline_label = 'AlphaVerus-Zero'
baseline_idx = next((i for i, row in enumerate(ablation_data) if row['Method'] == baseline_label), None)

print("\n--- Ablation Study Summary ---")
print("(Comparing ablations to AlphaVerus-Zero)\n")

# Print per-ablation results
for row in ablation_data:
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

# Compare ablations vs baseline
if baseline_idx is not None:
    print("\n--- Performance Drop vs AlphaVerus-Zero ---")

    baseline_row = ablation_data[baseline_idx]
    baseline_means = []
    for dataset in datasets:
        for metric in metrics:
            col_key = f"{dataset}/{metric}"
            mean_val = get_mean_from_tuple(baseline_row[col_key])
            if mean_val is not None:
                baseline_means.append(mean_val)

    baseline_avg = sum(baseline_means) / len(baseline_means) if baseline_means else 0

    for row in ablation_data:
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
                pct_drop = ((baseline_avg - avg) / baseline_avg) * 100
                print(f"  {method}: {avg*100:.2f}% ({diff_pp:+.2f}pp, {pct_drop:.1f}% drop)")

# ============================================================================
# VERIFICATION ABLATION P-VALUES (for verification_section.tex)
# ============================================================================
print("\n" + "=" * 80)
print("=== VERIFICATION ABLATION P-VALUES ===")
print("(For verification_section.tex - comparing AV0 vs verification ablations)")
print("=" * 80)


def get_raw_values(run_name: str, dataset: str, metric: str):
    """Get raw values for a specific run/dataset/metric."""
    _, _, raw_values = get_metric_mean_std_values(
        run_name=run_name,
        dataset=dataset,
        metric=metric,
        run_groups=run_groups
    )
    return raw_values


# Define comparisons for verification section
verification_comparisons = [
    ('AV0', 'AV0-solutionver', 'DAFNY2VERUS', 'pass@1', 'Dafny2Verus pass@1: AV0 vs -solution'),
    ('AV0', 'AV0-solutionver', 'DAFNY2VERUS', 'pass@5', 'Dafny2Verus pass@5: AV0 vs -solution'),
    ('AV0', 'AV0-solutionver', 'MBPP', 'pass@1', 'MBPP pass@1: AV0 vs -solution'),
    ('AV0', 'AV0-solutionver', 'MBPP', 'pass@5', 'MBPP pass@5: AV0 vs -solution'),
    ('AV0', 'AV0-solutionver', 'HUMANEVAL', 'pass@1', 'HUMANEVAL pass@1: AV0 vs -solution'),
    ('AV0', 'AV0-solutionver', 'HUMANEVAL', 'pass@5', 'HUMANEVAL pass@5: AV0 vs -solution'),
    ('AV0-solutionver', 'AV0-ver', 'DAFNY2VERUS', 'pass@1', 'Dafny2Verus pass@1: -solution vs -solution&spec'),
    ('AV0-solutionver', 'AV0-ver', 'DAFNY2VERUS', 'pass@5', 'Dafny2Verus pass@5: -solution vs -solution&spec'),
    ('AV0-solutionver', 'AV0-ver', 'MBPP', 'pass@1', 'MBPP pass@1: -solution vs -solution&spec'),
    ('AV0-solutionver', 'AV0-ver', 'MBPP', 'pass@5', 'MBPP pass@5: -solution vs -solution&spec'),
    ('AV0-solutionver', 'AV0-ver', 'HUMANEVAL', 'pass@1', 'HUMANEVAL pass@1: -solution vs -solution&spec'),
    ('AV0-solutionver', 'AV0-ver', 'HUMANEVAL', 'pass@5', 'HUMANEVAL pass@5: -solution vs -solution&spec'),
]

print("\nStatistical significance tests (paired t-test, one-sided: baseline > ablation):\n")
for baseline_run, ablation_run, dataset, metric, description in verification_comparisons:
    baseline_values = get_raw_values(baseline_run, dataset, metric)
    ablation_values = get_raw_values(ablation_run, dataset, metric)

    if baseline_values and ablation_values:
        # Test if baseline is significantly greater than ablation
        p_value = compute_pvalue(baseline_values, ablation_values, alternative='greater')
        baseline_mean = sum(baseline_values) / len(baseline_values)
        ablation_mean = sum(ablation_values) / len(ablation_values)
        rel_drop = ((baseline_mean - ablation_mean) / baseline_mean) * 100

        sig_marker = "***" if p_value and p_value < 0.001 else ("**" if p_value and p_value < 0.01 else ("*" if p_value and p_value < 0.05 else ""))
        p_str = f"p={p_value:.4f}" if p_value else "p=N/A"
        print(f"  {description}")
        print(f"    {baseline_mean*100:.2f}% -> {ablation_mean*100:.2f}% (drop: {rel_drop:.1f}%), {p_str} {sig_marker}")
    else:
        print(f"  {description}: Missing data")

print("\nLegend: *** p<0.001, ** p<0.01, * p<0.05")

# ============================================================================
# SAMPLING ABLATION: VALID SPECS AND TRAINING SAMPLES
# ============================================================================
print("\n" + "=" * 80)
print("=== SAMPLING ABLATION: VALID SPECS AND TRAINING SAMPLES ===")
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
        if get_base_run_name(run_name) != base_run_name:
            continue

        # Get the last iteration's data
        history = run_info.get('history', [])
        if not history:
            continue

        # Get the second-to-last entry (last entry often has None values)
        # Use index -2 if available, otherwise -1
        last_entry = history[-2] if len(history) >= 2 else history[-1]

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


# Get stats for AV0 and AV0-sampling2
av0_stats = get_proposal_stats('AV0')
sampling_stats = get_proposal_stats('AV0-sampling2')

print("\n--- AV0 (AlphaVerus-Zero) ---")
if av0_stats:
    for key, data in av0_stats.items():
        if data:
            print(f"  {key}: {data['mean']:.2f} (n={data['n']})")

print("\n--- AV0-sampling2 (-sampling ablation) ---")
if sampling_stats:
    for key, data in sampling_stats.items():
        if data:
            print(f"  {key}: {data['mean']:.2f} (n={data['n']})")

# Compare the two
print("\n--- Comparison (AV0 vs -sampling) ---")
if av0_stats and sampling_stats:
    for key in ['n_valid', 'n_unique', 'len_trn_data', 'uniqueness_rate']:
        av0_data = av0_stats.get(key)
        samp_data = sampling_stats.get(key)
        if av0_data and samp_data:
            if key == 'uniqueness_rate':
                print(f"  {key}: AV0={av0_data['mean']*100:.1f}%, -sampling={samp_data['mean']*100:.1f}%")
            else:
                ratio = av0_data['mean'] / samp_data['mean'] if samp_data['mean'] > 0 else float('inf')
                diff = av0_data['mean'] - samp_data['mean']
                print(f"  {key}: AV0={av0_data['mean']:.1f}, -sampling={samp_data['mean']:.1f} (AV0 is {ratio:.2f}x, +{diff:.1f})")

# ============================================================================
# SAMPLING ABLATION P-VALUES (for diversity_section.tex)
# ============================================================================
print("\n" + "=" * 80)
print("=== SAMPLING ABLATION P-VALUES ===")
print("(For diversity_section.tex - comparing AV0 vs -sampling)")
print("=" * 80)

# Define comparisons for sampling section
sampling_comparisons = [
    ('AV0', 'AV0-sampling2', 'DAFNY2VERUS', 'pass@1', 'Dafny2Verus pass@1'),
    ('AV0', 'AV0-sampling2', 'DAFNY2VERUS', 'pass@5', 'Dafny2Verus pass@5'),
    ('AV0', 'AV0-sampling2', 'DAFNY2VERUS', 'pass@10', 'Dafny2Verus pass@10'),
    ('AV0', 'AV0-sampling2', 'MBPP', 'pass@1', 'MBPP pass@1'),
    ('AV0', 'AV0-sampling2', 'MBPP', 'pass@5', 'MBPP pass@5'),
    ('AV0', 'AV0-sampling2', 'MBPP', 'pass@10', 'MBPP pass@10'),
    ('AV0', 'AV0-sampling2', 'HUMANEVAL', 'pass@1', 'HUMANEVAL pass@1'),
    ('AV0', 'AV0-sampling2', 'HUMANEVAL', 'pass@5', 'HUMANEVAL pass@5'),
    ('AV0', 'AV0-sampling2', 'HUMANEVAL', 'pass@10', 'HUMANEVAL pass@10'),
]

print("\nStatistical significance tests (paired t-test, one-sided: AV0 > -sampling):\n")
sig_count = 0
total_count = 0
for baseline_run, ablation_run, dataset, metric, description in sampling_comparisons:
    baseline_values = get_raw_values(baseline_run, dataset, metric)
    ablation_values = get_raw_values(ablation_run, dataset, metric)

    if baseline_values and ablation_values:
        total_count += 1
        # Test if baseline is significantly greater than ablation
        p_value = compute_pvalue(baseline_values, ablation_values, alternative='greater')
        baseline_mean = sum(baseline_values) / len(baseline_values)
        ablation_mean = sum(ablation_values) / len(ablation_values)
        rel_drop = ((baseline_mean - ablation_mean) / baseline_mean) * 100

        is_sig = p_value and p_value < 0.05
        if is_sig:
            sig_count += 1
        sig_marker = "***" if p_value and p_value < 0.001 else ("**" if p_value and p_value < 0.01 else ("*" if p_value and p_value < 0.05 else ""))
        p_str = f"p={p_value:.4f}" if p_value else "p=N/A"
        print(f"  {description}: {baseline_mean*100:.2f}% -> {ablation_mean*100:.2f}% (drop: {rel_drop:.1f}%), {p_str} {sig_marker}")
    else:
        print(f"  {description}: Missing data")

print(f"\nSummary: {sig_count} of {total_count} comparisons are statistically significant (p < 0.05)")
print("\nLegend: *** p<0.001, ** p<0.01, * p<0.05")

print("\n" + "=" * 80)
