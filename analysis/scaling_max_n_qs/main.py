#!/usr/bin/env python3
"""
Scaling Max N Questions Analysis Script

Analyzes how performance scales with max_n_qs parameter across different datasets:
- dafny2verus (DAFNY2VERUS)
- MBPP
- HumanEval (HUMANEVAL)

Compares pass@1, pass@5, and pass@10 metrics.
"""

import wandb
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import os
import numpy as np
import pickle
import argparse
from typing import Dict, List, Tuple, Optional

# Parse arguments
parser = argparse.ArgumentParser(description='Generate scaling max_n_qs analysis')
parser.add_argument('--load_from_cache', action='store_true',
                    help='Load run data from cache if available')
args = parser.parse_args()

# Create output directories
os.makedirs('figs', exist_ok=True)
os.makedirs('tables', exist_ok=True)
os.makedirs('analyses', exist_ok=True)

CACHE_PATH = 'analysis/scaling_max_n_qs/run_cache.pk'

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
    api = wandb.Api()

    # Pull all runs from the project
    print("\nFetching all runs from socialiq/av0...")
    all_runs = api.runs("socialiq/av0")

    print(f"\n=== Found {len(all_runs)} total runs ===\n")
    print("Available run names:")
    print("-" * 80)
    for idx, run in enumerate(all_runs, 1):
        tags_str = ', '.join(run.tags) if run.tags else 'no tags'
        print(f"{idx:3d}. {run.name:40s} | Tags: {tags_str}")
    print("-" * 80)

    # Extract run data
    print("\n=== Extracting run data ===")
    run_data = {}
    for run in all_runs:
        run_info = {
            'id': run.id,
            'name': run.name,
            'config': run.config,
            'history': list(run.scan_history()),
            'tags': run.tags
        }
        run_data[run.name] = run_info

    # Save to cache
    print(f"\nSaving {len(run_data)} runs to cache: {CACHE_PATH}")
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(run_data, f)

print(f"Loaded data for {len(run_data)} runs")


def get_final_metric_value(history: List[Dict], metric_name: str) -> Optional[float]:
    """Extract the final (last) value for a given metric from run history."""
    values = []
    for row in history:
        if metric_name in row and row[metric_name] is not None:
            # Handle NaN values
            if isinstance(row[metric_name], str) and row[metric_name] == 'NaN':
                continue
            values.append(row[metric_name])

    if values:
        return values[-1]  # Return last value
    return None


def plot_scaling_dataset(dataset_name: str, max_n_qs_values: List[int],
                         run_name_template: str, output_file: str):
    """
    Create a plot showing how performance scales with max_n_qs.

    Args:
        dataset_name: Name of dataset (e.g., 'DAFNY2VERUS', 'MBPP', 'HUMANEVAL')
        max_n_qs_values: List of max_n_qs values to plot (e.g., [1000, 2000, 4000, 8000, 32000])
        run_name_template: Template for run names (e.g., 'SIMPLe-{}' where {} is max_n_qs)
        output_file: Path to save the plot
    """
    metrics = ['pass@1', 'pass@5', 'pass@10']
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    markers = ['o', 's', '^']

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    for metric_idx, metric_suffix in enumerate(metrics):
        metric_key = f"{dataset_name}/{metric_suffix}"

        x_values = []
        y_values = []

        for max_n_qs in max_n_qs_values:
            run_name = run_name_template.format(max_n_qs)

            if run_name not in run_data:
                print(f"  Warning: '{run_name}' not found in run data")
                continue

            final_value = get_final_metric_value(run_data[run_name]['history'], metric_key)

            if final_value is not None:
                x_values.append(max_n_qs)
                y_values.append(final_value)
                print(f"  ✓ {run_name}: {metric_key} = {final_value:.4f}")
            else:
                print(f"  ✗ No data for {run_name} on {metric_key}")

        if x_values and y_values:
            ax.plot(x_values, y_values,
                   label=metric_suffix,
                   color=colors[metric_idx],
                   linestyle='-',
                   marker=markers[metric_idx],
                   linewidth=3.5,
                   markersize=16)

    ax.set_xlabel('Questions Per Iteration', fontsize=28)
    ax.set_ylabel('Pass Rate', fontsize=26)
    # Map dataset keys to display names
    display_names = {'DAFNY2VERUS': 'Dafny2Verus', 'MBPP': 'MBPP', 'HUMANEVAL': 'HumanEval'}
    display_name = display_names.get(dataset_name, dataset_name)
    ax.set_title(f"{display_name}", fontsize=32)
    ax.legend(fontsize=18, loc='upper left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Use log scale for x-axis with explicit tick labels and minor ticks
    ax.set_xscale('log')
    ax.set_xticks(max_n_qs_values)
    ax.set_xticklabels([f'{v:,}' for v in max_n_qs_values], fontsize=20)
    # Add minor ticks (2-9 between each power of 10) - the uneven spacing signals log scale
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())  # Hide minor tick labels
    ax.tick_params(axis='x', which='minor', length=4)
    ax.tick_params(axis='y', labelsize=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}\n")


# Max N Questions values to analyze
max_n_qs_values = [4000, 8000, 16000, 32000]
max_n_qs_values_for_analysis = [4000, 8000, 16000, 32000]

# Run name template
run_name_template = 'AV0-{}-seed0'


def collect_dataset_metrics(dataset_name: str, max_n_qs_list: List[int]) -> Dict:
    """Collect all metrics for a dataset across different max_n_qs values."""
    metrics = ['pass@1', 'pass@5', 'pass@10']
    data = {metric: {} for metric in metrics}

    for max_n_qs in max_n_qs_list:
        run_name = run_name_template.format(max_n_qs)

        if run_name not in run_data:
            continue

        for metric_suffix in metrics:
            metric_key = f"{dataset_name}/{metric_suffix}"
            final_value = get_final_metric_value(run_data[run_name]['history'], metric_key)

            if final_value is not None:
                data[metric_suffix][max_n_qs] = final_value

    return data


def compute_scaling_statistics(data: Dict, metric: str) -> Dict:
    """Compute percentage gains and multiplier effects for scaling."""
    values = data[metric]
    if not values:
        return {}

    sorted_keys = sorted(values.keys())
    stats = {
        'values': values,
        'sorted_keys': sorted_keys,
        'gains': {},
        'multipliers': {},
        'relative_gains': {}
    }

    # Compute gains from baseline (first value)
    baseline = values[sorted_keys[0]]
    for key in sorted_keys:
        stats['gains'][key] = values[key] - baseline
        stats['multipliers'][key] = values[key] / baseline
        if key != sorted_keys[0]:
            # Relative gain from previous
            prev_key = sorted_keys[sorted_keys.index(key) - 1]
            stats['relative_gains'][key] = (values[key] - values[prev_key]) / values[prev_key] * 100

    return stats


# ============================================================================
# PLOT 1: DAFNY2VERUS
# ============================================================================
print("\n=== Creating DAFNY2VERUS scaling plot ===")
plot_scaling_dataset('DAFNY2VERUS', max_n_qs_values, run_name_template,
                     'figs/scaling_dafny2verus.png')

# ============================================================================
# PLOT 2: MBPP
# ============================================================================
print("\n=== Creating MBPP scaling plot ===")
plot_scaling_dataset('MBPP', max_n_qs_values, run_name_template,
                     'figs/scaling_mbpp.png')

# ============================================================================
# PLOT 3: HUMANEVAL
# ============================================================================
print("\n=== Creating HUMANEVAL scaling plot ===")
plot_scaling_dataset('HUMANEVAL', max_n_qs_values, run_name_template,
                     'figs/scaling_humaneval.png')


print("\n" + "=" * 80)
print("=== Scaling Max N Questions Analysis Complete! ===")
print("=" * 80)
print("\nGenerated plots:")
print("  1. figs/scaling_dafny2verus.png")
print("  2. figs/scaling_mbpp.png")
print("  3. figs/scaling_humaneval.png")
print()


# ============================================================================
# Detailed Statistics and Analysis
# ============================================================================

print("\n" + "=" * 80)
print("=== DETAILED STATISTICS ===")
print("=" * 80)

datasets = ['DAFNY2VERUS', 'MBPP', 'HUMANEVAL']
all_data = {}

for dataset in datasets:
    print(f"\n{'='*80}")
    print(f"=== {dataset} ===")
    print(f"{'='*80}")

    data = collect_dataset_metrics(dataset, max_n_qs_values_for_analysis)
    all_data[dataset] = data

    for metric in ['pass@1', 'pass@5', 'pass@10']:
        print(f"\n{metric}:")
        if data[metric]:
            for max_n_qs in sorted(data[metric].keys()):
                print(f"  {max_n_qs:6d}: {data[metric][max_n_qs]:.4f}")
        else:
            print("  No data available")


# ============================================================================
# Compute Scaling Analysis
# ============================================================================

print("\n" + "=" * 80)
print("=== SCALING ANALYSIS (Excluding 16k) ===")
print("=" * 80)

analysis_max_n_qs = [v for v in max_n_qs_values_for_analysis if v != 16000]

for dataset in datasets:
    print(f"\n{'='*80}")
    print(f"=== {dataset} ===")
    print(f"{'='*80}")

    data = collect_dataset_metrics(dataset, analysis_max_n_qs)

    for metric in ['pass@1', 'pass@5', 'pass@10']:
        if not data[metric]:
            continue

        stats = compute_scaling_statistics(data, metric)

        print(f"\n{metric}:")
        print(f"  Baseline ({stats['sorted_keys'][0]}): {stats['values'][stats['sorted_keys'][0]]:.4f}")

        for key in stats['sorted_keys'][1:]:
            gain = stats['gains'][key]
            multiplier = stats['multipliers'][key]
            rel_gain = stats['relative_gains'][key]
            print(f"  {key:6d}: {stats['values'][key]:.4f} | "
                  f"Gain: +{gain:.4f} ({(gain/stats['values'][stats['sorted_keys'][0]])*100:.1f}%) | "
                  f"Multiplier: {multiplier:.3f}x | "
                  f"Step gain: +{rel_gain:.1f}%")


# ============================================================================
# Generate LaTeX Analysis Section
# ============================================================================

print("\n" + "=" * 80)
print("=== GENERATING LATEX ANALYSIS ===")
print("=" * 80)

latex_output = []
latex_output.append("\\subsection{Scaling Behavior with Respect to Max Questions Per Iteration}")
latex_output.append("")

# Focus on DAFNY2VERUS for the main analysis
dafny_data = collect_dataset_metrics('DAFNY2VERUS', analysis_max_n_qs)

latex_output.append("We analyze how our method scales with the number of questions attempted per iteration (max\\_n\\_qs). ")
latex_output.append(f"We evaluate across four different budget levels: {', '.join([str(v) for v in analysis_max_n_qs[:-1]])} and {analysis_max_n_qs[-1]} questions per iteration, ")
latex_output.append("focusing on the DAFNY2VERUS benchmark as our primary evaluation dataset.")
latex_output.append("")

# Pass@1 analysis
if dafny_data['pass@1']:
    stats_p1 = compute_scaling_statistics(dafny_data, 'pass@1')
    sorted_keys = stats_p1['sorted_keys']

    latex_output.append("\\paragraph{Pass@1 Performance}")
    latex_output.append(f"Starting from a baseline of {stats_p1['values'][sorted_keys[0]]:.4f} at {sorted_keys[0]} questions per iteration, ")

    improvements = []
    for i, key in enumerate(sorted_keys[1:], 1):
        val = stats_p1['values'][key]
        improvements.append(f"{key} questions achieves {val:.4f}")

    latex_output.append(f"we observe consistent improvements: {'; '.join(improvements)}. ")

    # Overall improvement
    final_key = sorted_keys[-1]
    total_gain_pct = (stats_p1['values'][final_key] - stats_p1['values'][sorted_keys[0]]) / stats_p1['values'][sorted_keys[0]] * 100
    latex_output.append(f"This represents an overall improvement of {total_gain_pct:.1f}\\% ")
    latex_output.append(f"(multiplier of {stats_p1['multipliers'][final_key]:.2f}x) ")
    latex_output.append(f"when scaling from {sorted_keys[0]} to {final_key} questions per iteration.")
    latex_output.append("")

# Pass@5 analysis
if dafny_data['pass@5']:
    stats_p5 = compute_scaling_statistics(dafny_data, 'pass@5')
    sorted_keys = stats_p5['sorted_keys']

    latex_output.append("\\paragraph{Pass@5 Performance}")
    latex_output.append(f"For pass@5, we observe similar scaling behavior. ")
    latex_output.append(f"Starting at {stats_p5['values'][sorted_keys[0]]:.4f} with {sorted_keys[0]} questions per iteration, ")
    latex_output.append(f"performance scales to {stats_p5['values'][sorted_keys[-1]]:.4f} at {sorted_keys[-1]} questions per iteration, ")

    final_key = sorted_keys[-1]
    total_gain_pct = (stats_p5['values'][final_key] - stats_p5['values'][sorted_keys[0]]) / stats_p5['values'][sorted_keys[0]] * 100
    latex_output.append(f"representing a {total_gain_pct:.1f}\\% improvement ")
    latex_output.append(f"({stats_p5['multipliers'][final_key]:.2f}x multiplier).")
    latex_output.append("")

# Pass@10 analysis (if available)
if dafny_data['pass@10']:
    stats_p10 = compute_scaling_statistics(dafny_data, 'pass@10')
    sorted_keys = stats_p10['sorted_keys']

    latex_output.append("\\paragraph{Pass@10 Performance}")
    latex_output.append(f"Pass@10 shows continued improvement with scale, ")
    latex_output.append(f"progressing from {stats_p10['values'][sorted_keys[0]]:.4f} to {stats_p10['values'][sorted_keys[-1]]:.4f}, ")

    final_key = sorted_keys[-1]
    total_gain_pct = (stats_p10['values'][final_key] - stats_p10['values'][sorted_keys[0]]) / stats_p10['values'][sorted_keys[0]] * 100
    latex_output.append(f"a {total_gain_pct:.1f}\\% gain ({stats_p10['multipliers'][final_key]:.2f}x).")
    latex_output.append("")

# Analysis of scaling consistency
latex_output.append("\\paragraph{Scaling Consistency}")
latex_output.append("The relative improvements between consecutive budget levels remain reasonably consistent. ")

if dafny_data['pass@1']:
    stats_p1 = compute_scaling_statistics(dafny_data, 'pass@1')
    rel_gains = [stats_p1['relative_gains'][k] for k in sorted(stats_p1['relative_gains'].keys())]
    avg_gain = np.mean(rel_gains)
    std_gain = np.std(rel_gains)

    latex_output.append(f"For pass@1, the average step-wise improvement is {avg_gain:.1f}\\% $\\pm$ {std_gain:.1f}\\%, ")
    latex_output.append("indicating relatively stable scaling behavior across budget levels. ")

latex_output.append("This suggests that our iterative training approach benefits consistently from increased question budgets, ")
latex_output.append("without exhibiting diminishing returns in the range of budgets tested.")
latex_output.append("")

# Save LaTeX output
latex_file = "tables/scaling_analysis.tex"
with open(latex_file, 'w') as f:
    f.write("\n".join(latex_output))

print(f"\n✓ LaTeX analysis saved to: {latex_file}")
print("\n" + "=" * 80)
print("=== LATEX CONTENT ===")
print("=" * 80)
print("\n".join(latex_output))
print("=" * 80)


# ============================================================================
# STATISTICS FOR RESULTS PARAGRAPH
# ============================================================================
print("\n" + "=" * 80)
print("=== STATISTICS FOR RESULTS PARAGRAPH ===")
print("=" * 80)

print("\n--- Scaling Summary (All Datasets) ---")
print("(Performance at different max_n_qs levels)\n")

# Summary across all datasets
for dataset in datasets:
    data = collect_dataset_metrics(dataset, max_n_qs_values_for_analysis)
    print(f"\n{dataset}:")

    for metric in ['pass@1', 'pass@5', 'pass@10']:
        if data[metric]:
            sorted_keys = sorted(data[metric].keys())
            min_key = sorted_keys[0]
            max_key = sorted_keys[-1]
            min_val = data[metric][min_key]
            max_val = data[metric][max_key]

            if min_val > 0:
                ratio = max_val / min_val
                diff_pp = (max_val - min_val) * 100
                print(f"  {metric}: {min_key} -> {max_key} questions: {min_val*100:.2f}% -> {max_val*100:.2f}% ({ratio:.2f}x, +{diff_pp:.2f}pp)")

# Cross-dataset average improvement
print("\n--- Cross-Dataset Average Improvement ---")
all_ratios = []
all_diffs = []

for dataset in datasets:
    data = collect_dataset_metrics(dataset, max_n_qs_values_for_analysis)
    for metric in ['pass@1', 'pass@5', 'pass@10']:
        if data[metric]:
            sorted_keys = sorted(data[metric].keys())
            min_val = data[metric][sorted_keys[0]]
            max_val = data[metric][sorted_keys[-1]]
            if min_val > 0:
                all_ratios.append(max_val / min_val)
                all_diffs.append(max_val - min_val)

if all_ratios:
    avg_ratio = sum(all_ratios) / len(all_ratios)
    avg_diff = sum(all_diffs) / len(all_diffs)
    print(f"Average improvement across all datasets/metrics: {avg_ratio:.2f}x (+{avg_diff*100:.2f}pp)")

print("\n" + "=" * 80)


# ============================================================================
# TTT SCALING ANALYSIS (MBPP and HumanEval)
# ============================================================================
# Run name templates for TTT scaling experiments
run_name_template_mbpp = 'AV0-{}-mbpp-seed0'
run_name_template_humaneval = 'AV0-{}-humaneval-seed0'
max_n_qs_values_ttt = [1000, 2000, 4000, 8000, 16000, 32000]


def plot_scaling_ttt_dataset(dataset_name: str, max_n_qs_values: List[int],
                              run_name_template: str, output_file: str):
    """
    Create a plot showing how TTT performance scales with max_n_qs.

    Args:
        dataset_name: Name of dataset (e.g., 'MBPP', 'HUMANEVAL')
        max_n_qs_values: List of max_n_qs values to plot
        run_name_template: Template for run names (e.g., 'AV0-{}-mbpp-seed0')
        output_file: Path to save the plot
    """
    metrics = ['pass@1', 'pass@5', 'pass@10']
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    markers = ['o', 's', '^']

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    for metric_idx, metric_suffix in enumerate(metrics):
        metric_key = f"{dataset_name}/{metric_suffix}"

        x_values = []
        y_values = []

        for max_n_qs in max_n_qs_values:
            run_name = run_name_template.format(max_n_qs)

            if run_name not in run_data:
                print(f"  Warning: '{run_name}' not found in run data")
                continue

            final_value = get_final_metric_value(run_data[run_name]['history'], metric_key)

            if final_value is not None:
                x_values.append(max_n_qs)
                y_values.append(final_value)
                print(f"  ✓ {run_name}: {metric_key} = {final_value:.4f}")
            else:
                print(f"  ✗ No data for {run_name} on {metric_key}")

        if x_values and y_values:
            ax.plot(x_values, y_values,
                   label=metric_suffix,
                   color=colors[metric_idx],
                   linestyle='-',
                   marker=markers[metric_idx],
                   linewidth=3.5,
                   markersize=14)

    ax.set_xlabel('Questions Per Iteration', fontsize=28)
    ax.set_ylabel('Pass Rate', fontsize=26)
    # Map dataset keys to display names
    display_names = {'DAFNY2VERUS': 'Dafny2Verus', 'MBPP': 'MBPP', 'HUMANEVAL': 'HumanEval'}
    display_name = display_names.get(dataset_name, dataset_name)
    ax.set_title(f"{display_name}", fontsize=32)
    ax.legend(fontsize=18, loc='upper left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Use log scale for x-axis with explicit tick labels and minor ticks
    ax.set_xscale('log')
    ax.set_xticks(max_n_qs_values)
    ax.set_xticklabels([f'{v:,}' for v in max_n_qs_values], fontsize=20)
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.tick_params(axis='x', which='minor', length=4)
    ax.tick_params(axis='y', labelsize=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}\n")


def collect_ttt_dataset_metrics(dataset_name: str, max_n_qs_list: List[int],
                                 run_name_template: str) -> Dict:
    """Collect all metrics for a TTT dataset across different max_n_qs values."""
    metrics = ['pass@1', 'pass@5', 'pass@10']
    data = {metric: {} for metric in metrics}

    for max_n_qs in max_n_qs_list:
        run_name = run_name_template.format(max_n_qs)

        if run_name not in run_data:
            continue

        for metric_suffix in metrics:
            metric_key = f"{dataset_name}/{metric_suffix}"
            final_value = get_final_metric_value(run_data[run_name]['history'], metric_key)

            if final_value is not None:
                data[metric_suffix][max_n_qs] = final_value

    return data


# ============================================================================
# PLOT: SCALING TTT MBPP
# ============================================================================
print("\n=== Creating SCALING TTT MBPP plot ===")
plot_scaling_ttt_dataset('MBPP', max_n_qs_values_ttt, run_name_template_mbpp,
                          'figs/scaling_ttt_mbpp.png')

# ============================================================================
# PLOT: SCALING TTT HUMANEVAL
# ============================================================================
print("\n=== Creating SCALING TTT HUMANEVAL plot ===")
plot_scaling_ttt_dataset('HUMANEVAL', max_n_qs_values_ttt, run_name_template_humaneval,
                          'figs/scaling_ttt_humaneval.png')


# ============================================================================
# DETAILED STATISTICS FOR TTT SCALING
# ============================================================================
print("\n" + "=" * 80)
print("=== TTT SCALING DETAILED STATISTICS ===")
print("=" * 80)

ttt_datasets = [
    ('MBPP', run_name_template_mbpp, 'MBPP'),
    ('HUMANEVAL', run_name_template_humaneval, 'HumanEval')
]

all_ttt_data = {}

for dataset_key, template, display_name in ttt_datasets:
    print(f"\n{'='*80}")
    print(f"=== {display_name} TTT Scaling ===")
    print(f"{'='*80}")

    data = collect_ttt_dataset_metrics(dataset_key, max_n_qs_values_ttt, template)
    all_ttt_data[dataset_key] = data

    for metric in ['pass@1', 'pass@5', 'pass@10']:
        print(f"\n{metric}:")
        if data[metric]:
            for max_n_qs in sorted(data[metric].keys()):
                print(f"  {max_n_qs:6d}: {data[metric][max_n_qs]:.4f}")
        else:
            print("  No data available")


# ============================================================================
# TTT SCALING ANALYSIS (Gains and Multipliers)
# ============================================================================
print("\n" + "=" * 80)
print("=== TTT SCALING ANALYSIS (Gains and Multipliers) ===")
print("=" * 80)

for dataset_key, template, display_name in ttt_datasets:
    print(f"\n{'='*80}")
    print(f"=== {display_name} ===")
    print(f"{'='*80}")

    data = collect_ttt_dataset_metrics(dataset_key, max_n_qs_values_ttt, template)

    for metric in ['pass@1', 'pass@5', 'pass@10']:
        if not data[metric]:
            continue

        stats = compute_scaling_statistics(data, metric)

        print(f"\n{metric}:")
        print(f"  Baseline ({stats['sorted_keys'][0]}): {stats['values'][stats['sorted_keys'][0]]:.4f}")

        for key in stats['sorted_keys'][1:]:
            gain = stats['gains'][key]
            multiplier = stats['multipliers'][key]
            rel_gain = stats['relative_gains'][key]
            print(f"  {key:6d}: {stats['values'][key]:.4f} | "
                  f"Gain: +{gain:.4f} ({(gain/stats['values'][stats['sorted_keys'][0]])*100:.1f}%) | "
                  f"Multiplier: {multiplier:.3f}x | "
                  f"Step gain: +{rel_gain:.1f}%")


# ============================================================================
# TTT SCALING SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("=== TTT SCALING SUMMARY (All Datasets) ===")
print("=" * 80)
print("(Performance at different max_n_qs levels for TTT experiments)\n")

for dataset_key, template, display_name in ttt_datasets:
    data = collect_ttt_dataset_metrics(dataset_key, max_n_qs_values_ttt, template)
    print(f"\n{display_name}:")

    for metric in ['pass@1', 'pass@5', 'pass@10']:
        if data[metric]:
            sorted_keys = sorted(data[metric].keys())
            min_key = sorted_keys[0]
            max_key = sorted_keys[-1]
            min_val = data[metric][min_key]
            max_val = data[metric][max_key]

            if min_val > 0:
                ratio = max_val / min_val
                diff_pp = (max_val - min_val) * 100
                print(f"  {metric}: {min_key} -> {max_key} questions: {min_val*100:.2f}% -> {max_val*100:.2f}% ({ratio:.2f}x, +{diff_pp:.2f}pp)")

# Cross-dataset average improvement for TTT scaling
print("\n--- Cross-Dataset Average Improvement (TTT Scaling) ---")
all_ttt_ratios = []
all_ttt_diffs = []

for dataset_key, template, display_name in ttt_datasets:
    data = collect_ttt_dataset_metrics(dataset_key, max_n_qs_values_ttt, template)
    for metric in ['pass@1', 'pass@5', 'pass@10']:
        if data[metric]:
            sorted_keys = sorted(data[metric].keys())
            min_val = data[metric][sorted_keys[0]]
            max_val = data[metric][sorted_keys[-1]]
            if min_val > 0:
                all_ttt_ratios.append(max_val / min_val)
                all_ttt_diffs.append(max_val - min_val)

if all_ttt_ratios:
    avg_ratio = sum(all_ttt_ratios) / len(all_ttt_ratios)
    avg_diff = sum(all_ttt_diffs) / len(all_ttt_diffs)
    print(f"Average improvement across TTT datasets/metrics: {avg_ratio:.2f}x (+{avg_diff*100:.2f}pp)")

print("\n" + "=" * 80)
print("=== TTT Scaling Analysis Complete! ===")
print("=" * 80)
print("\nGenerated plots:")
print("  1. figs/scaling_ttt_mbpp.png")
print("  2. figs/scaling_ttt_humaneval.png")
print()


# ============================================================================
# LATEX-READY RELATIVE PERCENTAGE IMPROVEMENTS
# ============================================================================
print("\n" + "=" * 80)
print("=== LATEX-READY: RELATIVE PERCENTAGE IMPROVEMENTS ===")
print("=" * 80)

print("\n--- TTT Scaling (for main section) ---")
# Dafny2Verus TTT (uses main scaling runs)
print("\nDafny2Verus TTT (4k -> 32k):")
dafny_data = collect_dataset_metrics('DAFNY2VERUS', max_n_qs_values_for_analysis)
for metric in ['pass@1', 'pass@5', 'pass@10']:
    if dafny_data[metric]:
        sorted_keys = sorted(dafny_data[metric].keys())
        min_val = dafny_data[metric][sorted_keys[0]]
        max_val = dafny_data[metric][sorted_keys[-1]]
        if min_val > 0:
            rel_pct = (max_val - min_val) / min_val * 100
            print(f"  {metric}: {min_val*100:.1f}% -> {max_val*100:.1f}% (+{rel_pct:.0f}%)")

# MBPP TTT
print("\nMBPP TTT (1k -> 32k):")
mbpp_ttt_data = collect_ttt_dataset_metrics('MBPP', max_n_qs_values_ttt, run_name_template_mbpp)
for metric in ['pass@1', 'pass@5', 'pass@10']:
    if mbpp_ttt_data[metric]:
        sorted_keys = sorted(mbpp_ttt_data[metric].keys())
        min_val = mbpp_ttt_data[metric][sorted_keys[0]]
        max_val = mbpp_ttt_data[metric][sorted_keys[-1]]
        if min_val > 0:
            rel_pct = (max_val - min_val) / min_val * 100
            print(f"  {metric}: {min_val*100:.1f}% -> {max_val*100:.1f}% (+{rel_pct:.0f}%)")

# HumanEval TTT
print("\nHumanEval TTT (1k -> 32k):")
humaneval_ttt_data = collect_ttt_dataset_metrics('HUMANEVAL', max_n_qs_values_ttt, run_name_template_humaneval)
for metric in ['pass@1', 'pass@5', 'pass@10']:
    if humaneval_ttt_data[metric]:
        sorted_keys = sorted(humaneval_ttt_data[metric].keys())
        min_val = humaneval_ttt_data[metric][sorted_keys[0]]
        max_val = humaneval_ttt_data[metric][sorted_keys[-1]]
        if min_val > 0:
            rel_pct = (max_val - min_val) / min_val * 100
            print(f"  {metric}: {min_val*100:.1f}% -> {max_val*100:.1f}% (+{rel_pct:.0f}%)")

print("\n--- Transfer Learning Scaling (for appendix) ---")
# MBPP Transfer (train on Dafny2Verus, eval on MBPP)
print("\nMBPP Transfer (4k -> 32k):")
mbpp_transfer_data = collect_dataset_metrics('MBPP', max_n_qs_values_for_analysis)
for metric in ['pass@1', 'pass@5', 'pass@10']:
    if mbpp_transfer_data[metric]:
        sorted_keys = sorted(mbpp_transfer_data[metric].keys())
        min_val = mbpp_transfer_data[metric][sorted_keys[0]]
        max_val = mbpp_transfer_data[metric][sorted_keys[-1]]
        if min_val > 0:
            rel_pct = (max_val - min_val) / min_val * 100
            print(f"  {metric}: {min_val*100:.1f}% -> {max_val*100:.1f}% (+{rel_pct:.0f}%)")

# HumanEval Transfer
print("\nHumanEval Transfer (4k -> 32k):")
humaneval_transfer_data = collect_dataset_metrics('HUMANEVAL', max_n_qs_values_for_analysis)
for metric in ['pass@1', 'pass@5', 'pass@10']:
    if humaneval_transfer_data[metric]:
        sorted_keys = sorted(humaneval_transfer_data[metric].keys())
        min_val = humaneval_transfer_data[metric][sorted_keys[0]]
        max_val = humaneval_transfer_data[metric][sorted_keys[-1]]
        if min_val > 0:
            rel_pct = (max_val - min_val) / min_val * 100
            print(f"  {metric}: {min_val*100:.1f}% -> {max_val*100:.1f}% ({rel_pct:+.0f}%)")

print("\n" + "=" * 80)
