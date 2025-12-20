#!/usr/bin/env python3
"""
Budget Analysis Script

Analyzes how performance scales with total question budget across iterations.
Run name pattern: AV0_Xi_bYk (e.g., AV0_1i_b4k, AV0_2i_b2k)
where:
  - Xi = iteration number (1i, 2i, 3i, ...)
  - bYk = total question budget (b1k=1000, b2k=2000, b4k=4000, etc.)
"""

import wandb
import matplotlib.pyplot as plt
import os
import re
import pickle
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Parse arguments
parser = argparse.ArgumentParser(description='Generate budget analysis plots and tables')
parser.add_argument('--load_from_cache', action='store_true',
                    help='Load run data from cache if available')
args = parser.parse_args()

# Create output directories
os.makedirs('figs', exist_ok=True)
os.makedirs('tables', exist_ok=True)
os.makedirs('analyses', exist_ok=True)

CACHE_PATH = 'analysis/budget/run_cache.pk'

# ============================================================================
# LOAD DATA FROM WANDB (or cache)
# ============================================================================
if args.load_from_cache and os.path.exists(CACHE_PATH):
    print("=== Loading from cache ===")
    with open(CACHE_PATH, 'rb') as f:
        all_runs_data = pickle.load(f)
    print(f"Loaded {len(all_runs_data)} runs from cache")
else:
    print("=== Connecting to W&B ===")
    api = wandb.Api()

    # Pull all runs from the project
    print("\nFetching all runs from socialiq/av0...")
    all_runs = api.runs("socialiq/av0")

    # Extract run data for caching
    all_runs_data = []
    for run in all_runs:
        all_runs_data.append({
            'name': run.name,
            'history': list(run.scan_history()),
            'tags': run.tags
        })

    # Save to cache
    print(f"\nSaving {len(all_runs_data)} runs to cache: {CACHE_PATH}")
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(all_runs_data, f)

print(f"\n=== Found {len(all_runs_data)} total runs ===\n")


def parse_run_name(name: str) -> Optional[Tuple[int, int]]:
    """
    Parse run name to extract iteration and budget.

    Expected format: AV0_Xi_bYk
    Examples:
      - AV0_1i_b4k -> (1, 4)
      - AV0_2i_b2k -> (2, 2)
      - AV0_3i_b1k -> (3, 1)

    Returns:
        (iteration, budget_k) or None if name doesn't match pattern
    """
    pattern = r'AV0_(\d+)i_b(\d+)k'
    match = re.match(pattern, name)
    if match:
        iteration = int(match.group(1))
        budget_k = int(match.group(2))
        return (iteration, budget_k)
    return None


def get_final_value(history: List[Dict], metric_name: str) -> Optional[float]:
    """
    Get the final (last logged) value for a metric.

    Args:
        history: List of history rows from W&B
        metric_name: Name of the metric (e.g., 'DAFNY2VERUS/pass@1')

    Returns:
        Final value or None if metric not found
    """
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


# Filter and parse runs
print("=== Filtering runs matching pattern AV0_Xi_bYk ===")
budget_runs = {}  # {run_name: {'iteration': int, 'budget_k': int, 'history': list}}

for run_info in all_runs_data:
    parsed = parse_run_name(run_info['name'])
    if parsed:
        iteration, budget_k = parsed
        print(f"  ✓ Found: {run_info['name']:30s} -> iteration={iteration}, budget={budget_k}k")
        budget_runs[run_info['name']] = {
            'iteration': iteration,
            'budget_k': budget_k,
            'history': run_info['history']
        }

print(f"\n=== Found {len(budget_runs)} matching runs ===\n")

if not budget_runs:
    print("ERROR: No runs matching pattern 'AV0_Xi_bYk' found!")
    print("\nAvailable run names:")
    for run_info in all_runs_data[:20]:
        print(f"  - {run_info['name']}")
    exit(1)

# Organize runs by budget and iteration
print("=== Organizing runs by budget and iteration ===")
data_by_budget = defaultdict(lambda: defaultdict(dict))  # {budget_k: {iteration: {metric: value}}}

metrics_to_extract = [
    'DAFNY2VERUS/pass@1',
    'DAFNY2VERUS/pass@5',
    'DAFNY2VERUS/pass@10',
    'MBPP/pass@1',
    'MBPP/pass@5',
    'HUMANEVAL/pass@1',
    'HUMANEVAL/pass@5'
]

for run_name, run_info in budget_runs.items():
    iteration = run_info['iteration']
    budget_k = run_info['budget_k']
    history = run_info['history']

    for metric in metrics_to_extract:
        value = get_final_value(history, metric)
        if value is not None:
            data_by_budget[budget_k][iteration][metric] = value
            print(f"  {run_name}: {metric} = {value:.4f}")

print(f"\n=== Extracted metrics for {len(data_by_budget)} budget levels ===")

# Get all unique budgets and iterations
all_budgets = sorted(data_by_budget.keys())
print(f"\nBudget levels found: {all_budgets} (in thousands)")


def plot_budget_comparison(dataset_name: str, metric_suffix: str, output_file: str):
    """
    Create a plot comparing different budget levels across iterations.

    Args:
        dataset_name: 'DAFNY2VERUS', 'MBPP', or 'HUMANEVAL'
        metric_suffix: 'pass@1' or 'pass@5'
        output_file: Path to save the plot
    """
    metric_key = f"{dataset_name}/{metric_suffix}"

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Define colors and markers for different budgets (matching other analysis scripts)
    budget_styles = {
        1: {'color': '#2E86AB', 'marker': 'o', 'linestyle': '-', 'label': '1k question budget'},
        2: {'color': '#A23B72', 'marker': 's', 'linestyle': '-', 'label': '2k question budget'},
        4: {'color': '#F18F01', 'marker': '^', 'linestyle': '-', 'label': '4k question budget'},
        8: {'color': '#96CEB4', 'marker': 'D', 'linestyle': '-', 'label': '8k question budget'},
        16: {'color': '#FF6B6B', 'marker': 'v', 'linestyle': '-', 'label': '16k question budget'},
    }

    plotted_any = False

    # Plot each budget level
    for budget_k in all_budgets:
        iterations = sorted(data_by_budget[budget_k].keys())
        values = []

        for iteration in iterations:
            if metric_key in data_by_budget[budget_k][iteration]:
                values.append(data_by_budget[budget_k][iteration][metric_key])
            else:
                values.append(None)

        # Filter out None values
        valid_iterations = [it for it, val in zip(iterations, values) if val is not None]
        valid_values = [val for val in values if val is not None]

        if valid_iterations and valid_values:
            style = budget_styles.get(budget_k, {
                'color': f'C{budget_k}',
                'marker': 'o',
                'linestyle': '-',
                'label': f'{budget_k}k question budget'
            })

            ax.plot(valid_iterations, valid_values,
                   label=style['label'],
                   color=style['color'],
                   marker=style['marker'],
                   linestyle=style['linestyle'],
                   linewidth=3.0,
                   markersize=14)
            plotted_any = True
            print(f"  ✓ Plotted {budget_k}k budget: {len(valid_iterations)} iterations")

    if plotted_any:
        ax.set_xlabel('# of iterations', fontsize=24)
        ax.set_ylabel(metric_suffix.replace('@', ' @ ').replace('pass', 'Pass'), fontsize=22)
        ax.legend(fontsize=15, loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=17)

        # Set integer ticks for iterations
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_file}\n")
    else:
        print(f"  ✗ No data to plot for {metric_key}\n")


# ============================================================================
# Generate all 6 plots
# ============================================================================

print("\n" + "=" * 80)
print("=== Generating Budget Comparison Plots ===")
print("=" * 80 + "\n")

# DAFNY2VERUS plots
print("=== DAFNY2VERUS pass@1 ===")
plot_budget_comparison('DAFNY2VERUS', 'pass@1', 'figs/budget_dafny2verus_pass1.png')

print("=== DAFNY2VERUS pass@5 ===")
plot_budget_comparison('DAFNY2VERUS', 'pass@5', 'figs/budget_dafny2verus_pass5.png')

# MBPP plots
print("=== MBPP pass@1 ===")
plot_budget_comparison('MBPP', 'pass@1', 'figs/budget_mbpp_pass1.png')

print("=== MBPP pass@5 ===")
plot_budget_comparison('MBPP', 'pass@5', 'figs/budget_mbpp_pass5.png')

# HUMANEVAL plots
print("=== HUMANEVAL pass@1 ===")
plot_budget_comparison('HUMANEVAL', 'pass@1', 'figs/budget_humaneval_pass1.png')

print("=== HUMANEVAL pass@5 ===")
plot_budget_comparison('HUMANEVAL', 'pass@5', 'figs/budget_humaneval_pass5.png')


print("\n" + "=" * 80)
print("=== Budget Analysis Complete! ===")
print("=" * 80)
print("\nGenerated plots:")
print("  1. figs/budget_dafny2verus_pass1.png")
print("  2. figs/budget_dafny2verus_pass5.png")
print("  3. figs/budget_mbpp_pass1.png")
print("  4. figs/budget_mbpp_pass5.png")
print("  5. figs/budget_humaneval_pass1.png")
print("  6. figs/budget_humaneval_pass5.png")
print()


# ============================================================================
# Print tables for each metric
# ============================================================================

def print_metric_table(metric_name: str, data_by_budget: Dict):
    """Print a formatted table for a specific metric."""
    print("\n" + "=" * 80)
    print(f"=== {metric_name} ===")
    print("=" * 80)

    # Get all unique iterations across all budgets
    all_iterations = set()
    for budget_k in data_by_budget:
        all_iterations.update(data_by_budget[budget_k].keys())
    all_iterations = sorted(all_iterations)

    # Get all budgets
    budgets = sorted(data_by_budget.keys())

    # Print header
    header = "Iteration"
    for budget_k in budgets:
        header += f" | {budget_k}k budget"
    print(header)
    print("-" * len(header))

    # Print rows
    for iteration in all_iterations:
        row = f"    {iteration}    "
        for budget_k in budgets:
            if iteration in data_by_budget[budget_k] and metric_name in data_by_budget[budget_k][iteration]:
                value = data_by_budget[budget_k][iteration][metric_name]
                row += f" |   {value:.4f}  "
            else:
                row += f" |     -     "
        print(row)
    print()


def generate_latex_table(metric_name: str, data_by_budget: Dict) -> str:
    """Generate a LaTeX table for a specific metric."""
    # Get all unique iterations across all budgets
    all_iterations = set()
    for budget_k in data_by_budget:
        all_iterations.update(data_by_budget[budget_k].keys())
    all_iterations = sorted(all_iterations)

    # Get all budgets
    budgets = sorted(data_by_budget.keys())

    # Start LaTeX table
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{c|" + "c" * len(budgets) + "}")
    latex.append("\\toprule")

    # Header
    header_row = "Iteration"
    for budget_k in budgets:
        header_row += f" & {budget_k}k budget"
    header_row += " \\\\"
    latex.append(header_row)
    latex.append("\\midrule")

    # Data rows
    for iteration in all_iterations:
        row = f"{iteration}"
        for budget_k in budgets:
            if iteration in data_by_budget[budget_k] and metric_name in data_by_budget[budget_k][iteration]:
                value = data_by_budget[budget_k][iteration][metric_name]
                row += f" & {value:.4f}"
            else:
                row += " & -"
        row += " \\\\"
        latex.append(row)

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append(f"\\caption{{{metric_name}}}")
    latex.append(f"\\label{{tab:{metric_name.lower().replace('/', '_').replace('@', '')}}}")
    latex.append("\\end{table}")

    return "\n".join(latex)


print("\n" + "=" * 80)
print("=== RESULTS TABLES ===")
print("=" * 80)

# Print tables for all metrics
for metric in metrics_to_extract:
    print_metric_table(metric, data_by_budget)

print("\n" + "=" * 80)
print("=== All Tables Generated! ===")
print("=" * 80)
print()


# ============================================================================
# Generate LaTeX tables
# ============================================================================

def generate_combined_latex_table(dataset_name: str, metrics: list, data_by_budget: Dict) -> str:
    """Generate a combined LaTeX table with multiple metrics for a single dataset."""
    # Get all unique iterations across all budgets
    all_iterations = set()
    for budget_k in data_by_budget:
        all_iterations.update(data_by_budget[budget_k].keys())
    all_iterations = sorted(all_iterations)

    # Get all budgets
    budgets = sorted(data_by_budget.keys())

    # Start LaTeX table
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")

    # Create column specification: iteration column + (pass@1, pass@5, pass@10) for each budget
    col_spec = "c|" + "|".join(["ccc"] * len(budgets))
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")

    # Multi-row header
    header_row1 = "Iteration"
    for budget_k in budgets:
        header_row1 += f" & \\multicolumn{{3}}{{c|}}{{{budget_k}k budget}}"
    header_row1 += " \\\\"
    latex.append(header_row1)

    # Sub-header with pass@k columns
    header_row2 = ""
    for budget_k in budgets:
        header_row2 += " & pass@1 & pass@5 & pass@10"
    header_row2 += " \\\\"
    latex.append(header_row2)
    latex.append("\\midrule")

    # Data rows
    for iteration in all_iterations:
        row = f"{iteration}"
        for budget_k in budgets:
            for metric_suffix in ["pass@1", "pass@5", "pass@10"]:
                metric_name = f"{dataset_name}/{metric_suffix}"
                if iteration in data_by_budget[budget_k] and metric_name in data_by_budget[budget_k][iteration]:
                    value = data_by_budget[budget_k][iteration][metric_name]
                    row += f" & {value:.4f}"
                else:
                    row += " & -"
        row += " \\\\"
        latex.append(row)

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append(f"\\caption{{{dataset_name} results across different budget levels and iterations}}")
    latex.append(f"\\label{{tab:{dataset_name.lower()}_combined}}")
    latex.append("\\end{table}")

    return "\n".join(latex)


print("\n" + "=" * 80)
print("=== Generating LaTeX Tables ===")
print("=" * 80)

latex_output = []
latex_output.append("% Budget Analysis Tables")
latex_output.append("% Generated automatically by analysis/budget/main.py")
latex_output.append("% Combined table with pass@1, pass@5, and pass@10 for DAFNY2VERUS")
latex_output.append("")

# Generate combined table for DAFNY2VERUS only
print(f"  ✓ Generating combined LaTeX table for DAFNY2VERUS")
dafny_metrics = ["pass@1", "pass@5", "pass@10"]
latex_output.append(generate_combined_latex_table("DAFNY2VERUS", dafny_metrics, data_by_budget))
latex_output.append("")

# Save to file
latex_file = "tables/budget.tex"
with open(latex_file, 'w') as f:
    f.write("\n".join(latex_output))

print(f"\n✓ LaTeX tables saved to: {latex_file}")
print("\n" + "=" * 80)
print("=== LaTeX Tables Complete! ===")
print("=" * 80)
print()


# ============================================================================
# STATISTICS FOR RESULTS PARAGRAPH
# ============================================================================
print("\n" + "=" * 80)
print("=== STATISTICS FOR RESULTS PARAGRAPH ===")
print("=" * 80)

print("\n--- Budget Analysis Statistics ---")
print("(Performance across different budget levels)\n")

# Get final iteration values for each budget
final_iteration = max(iteration for budget_k in data_by_budget for iteration in data_by_budget[budget_k])

print(f"Final iteration analyzed: {final_iteration}")

# Collect final values per budget for DAFNY2VERUS pass@1
print("\n--- Final Iteration Performance (DAFNY2VERUS pass@1) ---")
budget_final_values = {}
for budget_k in sorted(all_budgets):
    if final_iteration in data_by_budget[budget_k]:
        val = data_by_budget[budget_k][final_iteration].get('DAFNY2VERUS/pass@1')
        if val is not None:
            budget_final_values[budget_k] = val
            print(f"  {budget_k}k budget: {val*100:.2f}%")

# Compare budget levels
if budget_final_values:
    min_budget = min(budget_final_values.keys())
    max_budget = max(budget_final_values.keys())
    min_val = budget_final_values.get(min_budget)
    max_val = budget_final_values.get(max_budget)

    if min_val and max_val and min_val > 0:
        ratio = max_val / min_val
        diff_pp = (max_val - min_val) * 100
        print(f"\n  Improvement {min_budget}k -> {max_budget}k: {ratio:.2f}x (+{diff_pp:.2f}pp)")

# Best budget per dataset
print("\n--- Best Budget Level per Dataset (pass@1) ---")
for metric in ['DAFNY2VERUS/pass@1', 'MBPP/pass@1', 'HUMANEVAL/pass@1']:
    best_budget = None
    best_val = 0
    for budget_k in all_budgets:
        if final_iteration in data_by_budget[budget_k]:
            val = data_by_budget[budget_k][final_iteration].get(metric)
            if val is not None and val > best_val:
                best_val = val
                best_budget = budget_k
    if best_budget:
        dataset_name = metric.split('/')[0]
        print(f"  {dataset_name}: {best_budget}k budget ({best_val*100:.2f}%)")

# Marginal gains analysis
print("\n--- Marginal Gains Between Budget Levels (DAFNY2VERUS pass@1) ---")
sorted_budgets = sorted(budget_final_values.keys())
for i in range(1, len(sorted_budgets)):
    prev_budget = sorted_budgets[i-1]
    curr_budget = sorted_budgets[i]
    prev_val = budget_final_values.get(prev_budget)
    curr_val = budget_final_values.get(curr_budget)

    if prev_val and curr_val and prev_val > 0:
        gain = (curr_val - prev_val) / prev_val * 100
        budget_increase = curr_budget / prev_budget
        print(f"  {prev_budget}k -> {curr_budget}k ({budget_increase:.1f}x budget): +{gain:.1f}% gain")

print("\n" + "=" * 80)


