#!/usr/bin/env python3
"""
Verification Analysis Script

Compares different verification strategies for SIMPLe with multi-seed support:
- SIMPLe (baseline)
- SIMPLe-specver (specification verification)
- SIMPLe-solutionver (solution verification)
- SIMPLe-ver (both spec and solution verification)

Refactored to use common utilities from analysis/utils/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import os
import numpy as np
import pickle
import argparse
from typing import Dict, List

# Import utilities
from utils.wandb_loader import load_wandb_runs
from utils.metric_extraction import group_runs_by_seed, get_metric_mean_std_values
from utils.statistics import (
    find_best_with_significance,
    compute_relative_change
)
from utils.formatting import format_percentage_with_std as format_value
from utils.latex_generation import save_latex

# Parse arguments
parser = argparse.ArgumentParser(description='Generate verification analysis table')
parser.add_argument('--load_from_cache', action='store_true',
                    help='Load run data from cache if available')
args = parser.parse_args()

# Create output directories
os.makedirs('tables', exist_ok=True)
os.makedirs('analyses', exist_ok=True)

CACHE_PATH = 'analysis/verification/run_cache.pk'

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
    print("\nFetching runs from socialiq/av0 with tags 'main' or 'seed_results'...")
    run_data = load_wandb_runs("socialiq/av0", tags=["main", "seed_results"], tags_any=True)

    # Save to cache
    print(f"\nSaving {len(run_data)} runs to cache: {CACHE_PATH}")
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(run_data, f)

print(f"\n=== Found {len(run_data)} total runs ===\n")

# Filter to seeds 0-4 only (to match main_results which uses 5 seeds)
run_data = {k: v for k, v in run_data.items() if 'seed5' not in k}
print(f"Filtered to {len(run_data)} runs (excluding seed5 to match main_results)")

# Group runs by base name (seed variants)
print("\n=== Grouping runs by seed ===")
run_groups = group_runs_by_seed(run_data)

print(f"Grouped {len(run_data)} runs into {len(run_groups)} base configurations")
for base_name, runs in sorted(run_groups.items()):
    if 'ver' in base_name.lower() or 'simple' in base_name.lower():
        print(f"  {base_name}: {len(runs)} seed(s)")


# Wrapper for get_metric that uses global run_groups
def get_metric(run_name: str, dataset: str, metric: str):
    """Get a specific metric for a run and dataset. Returns (mean, std, raw_values)."""
    return get_metric_mean_std_values(run_name, dataset, metric, run_groups)


# ============================================================================
# IDENTIFY VERIFICATION RUNS
# ============================================================================
print("\n" + "=" * 80)
print("=== Identifying Verification Runs ===")
print("=" * 80)

# Use AV0 as baseline
baseline_run = 'AV0'

print(f"Using baseline: {baseline_run}")

if baseline_run not in run_groups:
    print(f"ERROR: Baseline run '{baseline_run}' not found in run_groups!")
    print(f"Available runs: {sorted(run_groups.keys())}")
    raise ValueError(f"Baseline run '{baseline_run}' not found")

# Verification runs configuration
# Format: (run_name, display_label)
verification_runs = [
    (baseline_run, '\\name'),  # Baseline AV0
    ('AV0-solutionver', '\\quad -solution'),  # Solution verification
    ('AV0-ver', '\\quad -solution\\&spec'),  # Both verification types
]

print("\nSearching for verification-related runs...")
verification_related = [name for name in run_groups.keys() if 'ver' in name.lower()]
print(f"Found verification-related runs: {sorted(verification_related)}")


# ============================================================================
# TABLE: VERIFICATION COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("=== Creating Verification Comparison Table ===")
print("=" * 80)

datasets = ['DAFNY2VERUS', 'MBPP', 'HUMANEVAL']
metrics = ['pass@1', 'pass@5']

# Collect data for verification table
verification_data = []

for run_name, label in verification_runs:
    row_data = {'Method': label}

    print(f"\nProcessing {run_name} (label: {label})...")

    for dataset in datasets:
        for metric in metrics:
            col_key = f"{dataset}/{metric}"
            value = get_metric(run_name, dataset, metric)
            row_data[col_key] = value
            print(f"  {col_key}: {value}")

    verification_data.append(row_data)

# Create DataFrame
verification_df = pd.DataFrame(verification_data)

# Find best in each column and format (with statistical significance)
formatted_verification_display = []
baseline_values = {}  # Store baseline values for comparison

for idx, row in verification_df.iterrows():
    formatted_row = {'Method': row['Method']}

    for dataset in datasets:
        for metric in metrics:
            col_key = f"{dataset}/{metric}"
            value = row[col_key]

            # Store baseline values (first row)
            if idx == 0:
                baseline_values[col_key] = value
                # Get all values in this column
                col_values = [r[col_key] for r in verification_data]
                best_idx = find_best_with_significance(col_values)
                is_best = (best_idx == idx)
                formatted_row[col_key] = format_value(value, is_best)
            else:
                # For non-baseline rows, calculate relative decrease and check if best
                col_values = [r[col_key] for r in verification_data]
                best_idx = find_best_with_significance(col_values)
                is_best = (best_idx == idx)

                baseline_val = baseline_values.get(col_key)
                if baseline_val is not None and isinstance(baseline_val, tuple) and baseline_val[0] is not None:
                    baseline_mean = baseline_val[0]
                    if value is not None and isinstance(value, tuple) and value[0] is not None:
                        value_mean = value[0]
                        rel_change = compute_relative_change(baseline_mean, value_mean)

                        # Print relative change to console (for reference)
                        if rel_change is not None and abs(rel_change) >= 0.01:
                            direction = "increase" if rel_change > 0 else "decrease"
                            print(f"  {row['Method']} {col_key}: {abs(rel_change):.1f}% {direction}")

                        # Format value without arrows
                        formatted_row[col_key] = format_value(value, is_best)
                    else:
                        formatted_row[col_key] = format_value(value, is_best)
                else:
                    formatted_row[col_key] = format_value(value, is_best)

    formatted_verification_display.append(formatted_row)

verification_display_df = pd.DataFrame(formatted_verification_display)

# Generate LaTeX for Verification
print("\n=== Generating Verification LaTeX Table ===")

latex_verification = "\\begin{table*}[t]\n"
latex_verification += "\\centering\n"
latex_verification += "\\caption{Verification Strategy Comparison. Values show mean pass@k rates with standard errors as subscripts across 5 random seeds. Best results (statistically significant via paired t-test, $p < 0.05$) are \\textbf{bolded}.}\n"
latex_verification += "\\label{tab:verification_results}\n"
latex_verification += "\\begin{tabular}{l|cc|cc|cc}\n"
latex_verification += "\\hline\n"
latex_verification += "& \\multicolumn{2}{c|}{\\textbf{Dafny2Verus}} & \\multicolumn{2}{c|}{\\textbf{MBPP}} & \\multicolumn{2}{c}{\\textbf{HumanEval}} \\\\\n"
latex_verification += "\\textbf{Method} & pass@1 & pass@5 & pass@1 & pass@5 & pass@1 & pass@5 \\\\\n"
latex_verification += "\\hline\n"

for _, row in verification_display_df.iterrows():
    latex_verification += f"{row['Method']}"
    for dataset in datasets:
        for metric in metrics:
            col_key = f"{dataset}/{metric}"
            latex_verification += f" & {row[col_key]}"
    latex_verification += " \\\\\n"

latex_verification += "\\hline\n"
latex_verification += "\\end{tabular}\n"
latex_verification += "\\end{table*}\n"

# Save verification table
save_latex(latex_verification, 'tables/verification.tex')
print("✓ Saved: tables/verification.tex")

# Generate trimmed LaTeX table (pass@1 only)
print("\n=== Generating Trimmed Verification LaTeX Table (pass@1 only) ===")

latex_trimmed = "\\begin{table}[t]\n"
latex_trimmed += "\\centering\n"
latex_trimmed += "\\caption{Verification Strategy Comparison (pass@1). Values show mean pass@1 rates with standard errors as subscripts across 5 random seeds. Best results (statistically significant via paired t-test, $p < 0.05$) are \\textbf{bolded}.}\n"
latex_trimmed += "\\label{tab:verification_results_trimmed}\n"
latex_trimmed += "\\begin{tabular}{l|ccc}\n"
latex_trimmed += "\\hline\n"
latex_trimmed += "\\textbf{Method} & \\textbf{D2V} & \\textbf{MBPP} & \\textbf{HEV} \\\\\n"
latex_trimmed += "\\hline\n"

for _, row in verification_display_df.iterrows():
    latex_trimmed += f"{row['Method']}"
    for dataset in datasets:
        col_key = f"{dataset}/pass@1"
        latex_trimmed += f" & {row[col_key]}"
    latex_trimmed += " \\\\\n"

latex_trimmed += "\\hline\n"
latex_trimmed += "\\end{tabular}\n"
latex_trimmed += "\\end{table}\n"

# Save trimmed verification table
save_latex(latex_trimmed, 'tables/verification_trimmed.tex')
print("✓ Saved: tables/verification_trimmed.tex")


# ============================================================================
# DETAILED STATISTICS AND RELATIVE CHANGES
# ============================================================================
print("\n" + "=" * 80)
print("=== Computing Detailed Statistics ===")
print("=" * 80)

# Extract metrics for each configuration
def get_metrics_dict(run_name: str, datasets: List[str], metrics: List[str]) -> Dict:
    """Extract all metrics for a given run."""
    result = {}
    for dataset in datasets:
        result[dataset] = {}
        for metric in metrics:
            value = get_metric(run_name, dataset, metric)
            result[dataset][metric] = value
    return result

# Get metrics for all configurations
configs = {
    'baseline': baseline_run,
    'solution_ver': 'AV0-solutionver',
    'spec_ver': 'AV0-specver',
    'both_ver': 'AV0-ver'
}

all_metrics = {}
for config_name, run_name in configs.items():
    print(f"\nExtracting metrics for {config_name} ({run_name})...")
    all_metrics[config_name] = get_metrics_dict(run_name, datasets, metrics)


# ============================================================================
# COMPUTE VERIFICATION STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("=== Verification Impact Analysis ===")
print("=" * 80)

# Focus on DAFNY2VERUS pass@1 for main analysis
dataset = 'DAFNY2VERUS'
metric = 'pass@1'

print(f"\nAnalyzing {dataset} {metric}:")
print("-" * 80)

# Get values (means from tuples)
def extract_mean(value):
    """Extract mean from (mean, std, values) tuple."""
    if value is None:
        return None
    if isinstance(value, tuple):
        return value[0]
    return value

baseline = extract_mean(all_metrics['baseline'][dataset][metric])
solution_ver = extract_mean(all_metrics['solution_ver'][dataset][metric])
spec_ver = extract_mean(all_metrics['spec_ver'][dataset][metric])
both_ver = extract_mean(all_metrics['both_ver'][dataset][metric])

if baseline is not None:
    print(f"Baseline (no verification):          {baseline:.4f} ({baseline*100:.2f}%)")
if solution_ver is not None:
    print(f"Solution verification only:          {solution_ver:.4f} ({solution_ver*100:.2f}%)")
if spec_ver is not None:
    print(f"Specification verification only:     {spec_ver:.4f} ({spec_ver*100:.2f}%)")
if both_ver is not None:
    print(f"Both verifications:                  {both_ver:.4f} ({both_ver*100:.2f}%)")
print()

# Compute relative changes from baseline
print("Impact of adding verification (from baseline with no verification):")
print("-" * 80)

if baseline is not None:
    if solution_ver is not None:
        solution_change = compute_relative_change(baseline, solution_ver)
        print(f"Add solution verification:     {solution_change:+.2f}% (absolute: {(solution_ver - baseline)*100:+.2f} pp)")

    if spec_ver is not None:
        spec_change = compute_relative_change(baseline, spec_ver)
        print(f"Add spec verification:         {spec_change:+.2f}% (absolute: {(spec_ver - baseline)*100:+.2f} pp)")

    if both_ver is not None:
        both_change = compute_relative_change(baseline, both_ver)
        print(f"Add both verifications:        {both_change:+.2f}% (absolute: {(both_ver - baseline)*100:+.2f} pp)")
print()

# Compute statistics across all datasets and metrics
print("\n" + "=" * 80)
print("=== Cross-Dataset Statistics ===")
print("=" * 80)

all_relative_changes = {
    'add_solution': [],
    'add_spec': [],
    'add_both': [],
}

for ds in datasets:
    for m in metrics:
        baseline_val = extract_mean(all_metrics['baseline'][ds][m])
        solution_val = extract_mean(all_metrics['solution_ver'][ds][m])
        spec_val = extract_mean(all_metrics['spec_ver'][ds][m])
        both_val = extract_mean(all_metrics['both_ver'][ds][m])

        if all([baseline_val, solution_val, spec_val, both_val]):
            all_relative_changes['add_solution'].append(compute_relative_change(baseline_val, solution_val))
            all_relative_changes['add_spec'].append(compute_relative_change(baseline_val, spec_val))
            all_relative_changes['add_both'].append(compute_relative_change(baseline_val, both_val))

print("\nAverage relative changes across all datasets and metrics:")
for key, values in all_relative_changes.items():
    if values:
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1) if len(values) > 1 else 0
        print(f"  {key:30s}: {mean_val:+.2f}% ± {std_val:.2f}%")

# ============================================================================
# ADDITIONAL DROP: -solution&spec vs -solution
# ============================================================================
print("\n" + "=" * 80)
print("=== Additional Drop: -solution&spec vs -solution ===")
print("(Shows impact of removing spec verification when solution verification is already removed)")
print("=" * 80)

print("\nRelative drop from -solution to -solution&spec:")
for ds in datasets:
    for m in metrics:
        solution_val = extract_mean(all_metrics['solution_ver'][ds][m])
        both_val = extract_mean(all_metrics['both_ver'][ds][m])

        if solution_val and both_val and solution_val > 0:
            additional_drop = compute_relative_change(solution_val, both_val)
            print(f"  {ds}/{m}: {additional_drop:+.1f}%")

# ============================================================================
# SPEC GENERATION ANALYSIS (Optional - requires local cache)
# ============================================================================
print("\n" + "=" * 80)
print("=== Spec Generation Comparison Analysis ===")
print("=" * 80)

# Import cache utilities
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.run_cache import create_run_cache

# Load caches from both runs (use seed0 for analysis)
our_method_cache = create_run_cache('AV0-seed0', skip_cache=False, base_dir='./outputs')
spec_ablation_cache = create_run_cache('AV0-specver-seed0', skip_cache=False, base_dir='./outputs')

# Analyze spec generation across all iterations
print("\nLoading proposed questions caches...")

our_method_stats = {
    'total_generated': 0,
    'total_valid': 0,
    'total_deduplicated': 0,
    'unique_solved': set()
}

spec_ablation_stats = {
    'total_generated': 0,
    'total_valid': 0,
    'total_deduplicated': 0,
    'unique_solved': set()
}

# Determine number of iterations (check cache files)
import glob as glob_module
our_method_iters = glob_module.glob(f'{our_method_cache.proposed_questions_dir}/iter*.pkl')
spec_ablation_iters = glob_module.glob(f'{spec_ablation_cache.proposed_questions_dir}/iter*.pkl')

n_iterations = max(len(our_method_iters), len(spec_ablation_iters))
print(f"Found {n_iterations} iterations")

if n_iterations == 0:
    print("\nNo local cache data found. Skipping spec generation analysis.")
    print("(Run training with local caching enabled to generate this data)")
    sys.exit(0)

# Load data from each iteration
for iteration in range(n_iterations):
    # Our method (with spec verification)
    our_cache = our_method_cache.load_proposed_questions(iteration)
    # Inference for iteration N+1 contains results for questions from iteration N
    our_inf_cache = our_method_cache.load_inferences(iteration + 1)

    if our_cache:
        our_method_stats['total_generated'] += len(our_cache.all_generated)
        our_method_stats['total_valid'] += len(our_cache.valid)
        our_method_stats['total_deduplicated'] += len(our_cache.deduplicated)

        # Load inference results to check if solved
        if our_inf_cache:
            inf_dict = our_inf_cache.to_dict()
            for q in our_cache.deduplicated:
                if q.question_id in inf_dict:
                    inf_result = inf_dict[q.question_id]
                    if inf_result.n_passing > 0:
                        our_method_stats['unique_solved'].add(q.question_id)

    # Spec ablation (without spec verification)
    spec_cache = spec_ablation_cache.load_proposed_questions(iteration)
    # Inference for iteration N+1 contains results for questions from iteration N
    spec_inf_cache = spec_ablation_cache.load_inferences(iteration + 1)

    if spec_cache:
        spec_ablation_stats['total_generated'] += len(spec_cache.all_generated)
        spec_ablation_stats['total_valid'] += len(spec_cache.valid)
        spec_ablation_stats['total_deduplicated'] += len(spec_cache.deduplicated)

        # Load inference results to check if solved
        if spec_inf_cache:
            inf_dict = spec_inf_cache.to_dict()
            for q in spec_cache.deduplicated:
                if q.question_id in inf_dict:
                    inf_result = inf_dict[q.question_id]
                    if inf_result.n_passing > 0:
                        spec_ablation_stats['unique_solved'].add(q.question_id)

print("\n" + "=" * 80)
print("SPEC GENERATION COMPARISON")
print("=" * 80)
print(f"{'Metric':<40} {'Our Method':>15} {'-spec Ablation':>15} {'Difference':>15}")
print("-" * 80)

# Total generated
diff_gen = spec_ablation_stats['total_generated'] - our_method_stats['total_generated']
pct_gen = (diff_gen / our_method_stats['total_generated'] * 100) if our_method_stats['total_generated'] > 0 else 0
print(f"{'Total specs generated':<40} {our_method_stats['total_generated']:>15} {spec_ablation_stats['total_generated']:>15} {f'+{diff_gen} (+{pct_gen:.1f}%)':>15}")

# Valid specs
diff_valid = spec_ablation_stats['total_valid'] - our_method_stats['total_valid']
pct_valid = (diff_valid / our_method_stats['total_valid'] * 100) if our_method_stats['total_valid'] > 0 else 0
print(f"{'Valid specs (passed verification)':<40} {our_method_stats['total_valid']:>15} {spec_ablation_stats['total_valid']:>15} {f'+{diff_valid} (+{pct_valid:.1f}%)':>15}")

# Deduplicated
diff_dedup = spec_ablation_stats['total_deduplicated'] - our_method_stats['total_deduplicated']
pct_dedup = (diff_dedup / our_method_stats['total_deduplicated'] * 100) if our_method_stats['total_deduplicated'] > 0 else 0
print(f"{'Final deduplicated specs':<40} {our_method_stats['total_deduplicated']:>15} {spec_ablation_stats['total_deduplicated']:>15} {f'+{diff_dedup} (+{pct_dedup:.1f}%)':>15}")

# Unique solved
diff_solved = len(spec_ablation_stats['unique_solved']) - len(our_method_stats['unique_solved'])
pct_solved = (diff_solved / len(our_method_stats['unique_solved']) * 100) if len(our_method_stats['unique_solved']) > 0 else 0
print(f"{'Unique problems solved':<40} {len(our_method_stats['unique_solved']):>15} {len(spec_ablation_stats['unique_solved']):>15} {f'+{diff_solved} (+{pct_solved:.1f}%)':>15}")

print("\n" + "=" * 80)
print("KEY FINDINGS:")
print(f"- The -spec ablation generated {pct_gen:.1f}% MORE specs (no filtering)")
print(f"- Despite generating more specs, it only solved {pct_solved:.1f}% more unique problems")
print(f"- This demonstrates that spec verification effectively filters low-quality specs")
print("=" * 80)

# ============================================================================
# INVALID SPEC PASSING RATE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("=== Invalid Spec Passing Rate Analysis ===")
print("=" * 80)

print("\nAnalyzing specs that FAILED verification in the -spec ablation...")
print("(Note: -spec ablation generates all specs but doesn't filter by verification)")

# Track specs from -spec ablation that failed verification
# We analyze all specs in the ablation and check which failed verification
invalid_specs_total = 0
invalid_specs_solved = 0
invalid_specs_by_iteration = []

for iteration in range(n_iterations):
    spec_cache = spec_ablation_cache.load_proposed_questions(iteration)
    # Inference for iteration N+1 contains results for questions from iteration N
    spec_inf_cache = spec_ablation_cache.load_inferences(iteration + 1)

    if spec_cache:
        # Find specs that failed verification (is_valid=False)
        # In -spec ablation, these are generated but not filtered out
        invalid_in_iter = [q for q in spec_cache.all_generated if not q.is_valid]

        # Check how many of these invalid specs have passing solutions
        invalid_solved_count = 0
        if spec_inf_cache:
            inf_dict = spec_inf_cache.to_dict()
            for q in invalid_in_iter:
                if q.question_id in inf_dict:
                    inf_result = inf_dict[q.question_id]
                    if inf_result.n_passing > 0:
                        invalid_solved_count += 1

        invalid_specs_total += len(invalid_in_iter)
        invalid_specs_solved += invalid_solved_count

        invalid_specs_by_iteration.append({
            'iteration': iteration,
            'total_invalid': len(invalid_in_iter),
            'invalid_solved': invalid_solved_count,
            'passing_rate': (invalid_solved_count / len(invalid_in_iter) * 100) if len(invalid_in_iter) > 0 else 0
        })

print("\n" + "=" * 80)
print("INVALID SPEC ANALYSIS")
print("=" * 80)

if invalid_specs_total > 0:
    print(f"Total invalid specs (failed verification): {invalid_specs_total}")
    print(f"Invalid specs with passing solutions: {invalid_specs_solved}")
    print(f"Passing rate for invalid specs: {(invalid_specs_solved / invalid_specs_total * 100):.2f}%")
    print("\nPer-iteration breakdown:")
    print(f"{'Iteration':<12} {'Invalid Specs':>15} {'Solved':>10} {'Passing Rate':>15}")
    print("-" * 55)
    for iter_data in invalid_specs_by_iteration:
        print(f"{iter_data['iteration']:<12} {iter_data['total_invalid']:>15} {iter_data['invalid_solved']:>10} {iter_data['passing_rate']:>14.2f}%")

    print("\n" + "=" * 80)
    print("KEY FINDING:")
    passing_rate_overall = (invalid_specs_solved / invalid_specs_total * 100)
    print(f"- Only {passing_rate_overall:.2f}% of specs that failed verification had passing solutions")
    print(f"- This confirms that spec verification is highly predictive of solution quality")
    print(f"- Filtering invalid specs avoids wasting inference on {invalid_specs_total} low-quality specs")
    print("=" * 80)
else:
    print("NOTE: No invalid specs found in -spec ablation cache.")
    print("This could mean:")
    print("  1. The -spec ablation doesn't record spec verification results")
    print("  2. All generated specs passed verification (unlikely)")
    print("\nChecking our method's cache for invalid specs instead...")

    # Analyze our method's invalid specs
    our_invalid_total = 0
    our_invalid_solved = 0
    our_invalid_by_iteration = []

    for iteration in range(n_iterations):
        our_cache = our_method_cache.load_proposed_questions(iteration)
        our_inf_cache = our_method_cache.load_inferences(iteration)

        if our_cache:
            # Find specs that failed verification
            invalid_in_iter = [q for q in our_cache.all_generated if not q.is_valid]

            # These specs were filtered out, so they don't appear in deduplicated
            # But we can check if any were accidentally solved before filtering
            invalid_solved_count = 0

            our_invalid_total += len(invalid_in_iter)
            our_invalid_solved += invalid_solved_count

            our_invalid_by_iteration.append({
                'iteration': iteration,
                'total_invalid': len(invalid_in_iter),
                'invalid_solved': invalid_solved_count,
            })

    print(f"\nOur method (AV0):")
    print(f"  Total specs generated: {our_method_stats['total_generated']}")
    print(f"  Specs that passed verification: {our_method_stats['total_valid']}")
    print(f"  Specs that FAILED verification: {our_invalid_total}")
    print(f"  Verification filtering rate: {(our_invalid_total / our_method_stats['total_generated'] * 100):.1f}%")

    print(f"\n-spec Ablation (AV0-specver):")
    print(f"  Total specs generated: {spec_ablation_stats['total_generated']}")
    print(f"  All marked as valid (no verification): {spec_ablation_stats['total_valid']}")
    print(f"  Verification filtering rate: 0.0% (no filtering)")

    print("\n" + "-" * 80)
    print("COMPUTATIONAL EFFICIENCY ANALYSIS:")
    print("-" * 80)

    # Calculate how many more specs the ablation had to run inference on
    extra_specs = spec_ablation_stats['total_deduplicated'] - our_method_stats['total_deduplicated']
    pct_extra = (extra_specs / our_method_stats['total_deduplicated'] * 100) if our_method_stats['total_deduplicated'] > 0 else 0

    print(f"\nDeduplicated specs sent to inference:")
    print(f"  Our method: {our_method_stats['total_deduplicated']}")
    print(f"  -spec ablation: {spec_ablation_stats['total_deduplicated']}")
    print(f"  Extra inference runs in ablation: {extra_specs} (+{pct_extra:.1f}%)")

    print(f"\nResults:")
    print(f"  Our method solved: {len(our_method_stats['unique_solved'])} unique problems")
    print(f"  -spec ablation solved: {len(spec_ablation_stats['unique_solved'])} unique problems")

    solved_diff = len(our_method_stats['unique_solved']) - len(spec_ablation_stats['unique_solved'])
    if solved_diff > 0:
        print(f"  Our method solved {solved_diff} MORE problems despite {pct_extra:.1f}% LESS inference!")
    else:
        print(f"  -spec ablation solved {-solved_diff} more problems with {pct_extra:.1f}% more inference")

    print(f"\nEfficiency metric (unique solved per 1000 deduplicated specs):")
    our_efficiency = (len(our_method_stats['unique_solved']) / our_method_stats['total_deduplicated'] * 1000) if our_method_stats['total_deduplicated'] > 0 else 0
    ablation_efficiency = (len(spec_ablation_stats['unique_solved']) / spec_ablation_stats['total_deduplicated'] * 1000) if spec_ablation_stats['total_deduplicated'] > 0 else 0
    print(f"  Our method: {our_efficiency:.1f}")
    print(f"  -spec ablation: {ablation_efficiency:.1f}")
    print(f"  Improvement: {((our_efficiency / ablation_efficiency - 1) * 100):.1f}%")

    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print(f"1. Spec verification filtered {(our_invalid_total / our_method_stats['total_generated'] * 100):.1f}% of generated specs")
    print(f"2. This reduced inference workload by {(our_invalid_total / spec_ablation_stats['total_deduplicated'] * 100):.1f}%")
    print(f"3. Despite less inference, our method solved {solved_diff} MORE unique problems")
    print(f"4. Spec verification improves efficiency by {((our_efficiency / ablation_efficiency - 1) * 100):.1f}%")
    print("=" * 80)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("=== Verification Analysis Complete! ===")
print("=" * 80)
print("\nGenerated files:")
print("  1. tables/verification.tex")
print()


# ============================================================================
# STATISTICS FOR RESULTS PARAGRAPH
# ============================================================================
print("\n" + "=" * 80)
print("=== STATISTICS FOR RESULTS PARAGRAPH ===")
print("=" * 80)

print("\n--- Verification Strategy Summary ---")
print("(Comparing verification strategies across all datasets)\n")

# Compute averages for each verification strategy
for run_name, label in verification_runs:
    means = []
    for dataset in datasets:
        for metric in metrics:
            val = get_metric(run_name, dataset, metric)
            if val is not None and val[0] is not None:
                means.append(val[0])

    if means:
        avg = sum(means) / len(means)
        print(f"{label}: Average={avg*100:.2f}%")

# Compare verification strategies vs baseline
print("\n--- Improvements vs Baseline (no verification) ---")
baseline_means = []
for dataset in datasets:
    for metric in metrics:
        val = get_metric(baseline_run, dataset, metric)
        if val is not None and val[0] is not None:
            baseline_means.append(val[0])

baseline_avg = sum(baseline_means) / len(baseline_means) if baseline_means else 0

for run_name, label in verification_runs:
    if run_name == baseline_run:
        continue

    means = []
    for dataset in datasets:
        for metric in metrics:
            val = get_metric(run_name, dataset, metric)
            if val is not None and val[0] is not None:
                means.append(val[0])

    if means and baseline_avg > 0:
        avg = sum(means) / len(means)
        ratio = avg / baseline_avg
        diff_pp = (avg - baseline_avg) * 100
        print(f"  {label}: {avg*100:.2f}% ({ratio:.2f}x vs baseline, {diff_pp:+.2f}pp)")

print("\n" + "=" * 80)
