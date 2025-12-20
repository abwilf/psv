"""
LaTeX generation utilities for analysis scripts.

Provides functions for creating LaTeX table headers, footers, multicolumn headers,
and saving LaTeX content to files.
"""

from typing import List


def create_latex_table_header(caption: str, label: str, tabular_spec: str) -> str:
    """
    Create LaTeX table header with caption, label, and tabular spec.

    Args:
        caption: Table caption text
        label: Table label for referencing (e.g., "tab:results")
        tabular_spec: Tabular column specification (e.g., "l|cc|cc")

    Returns:
        LaTeX header string

    Example:
        >>> create_latex_table_header("Results", "tab:results", "l|cc")
        '\\begin{table*}[t]\\n\\centering\\n...'
    """
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{tabular_spec}}}",
        "\\hline"
    ]
    return "\n".join(lines)


def create_latex_table_footer() -> str:
    """
    Create LaTeX table footer.

    Returns:
        LaTeX footer string

    Example:
        >>> create_latex_table_footer()
        '\\hline\\n\\end{tabular}\\n\\end{table*}'
    """
    lines = [
        "\\hline",
        "\\end{tabular}",
        "\\end{table*}"
    ]
    return "\n".join(lines)


def format_multicolumn_header(datasets: List[str], metrics: List[str]) -> str:
    """
    Format multicolumn header for datasets and metrics.

    Args:
        datasets: List of dataset names (e.g., ['DAFNY2VERUS', 'MBPP'])
        metrics: List of metric names (e.g., ['pass@1', 'pass@5'])

    Returns:
        Formatted multicolumn header string (two lines)

    Example:
        >>> format_multicolumn_header(['DS1', 'DS2'], ['m1', 'm2'])
        '& \\multicolumn{2}{c|}{\\textbf{Ds1}} & \\multicolumn{2}{c}{\\textbf{Ds2}} \\\\\\n...'
    """
    # Dataset row
    dataset_parts = []
    for i, dataset in enumerate(datasets):
        # Format dataset name (special handling for known datasets)
        if dataset == 'DAFNY2VERUS':
            display_name = 'Dafny2Verus'
        elif dataset == 'HUMANEVAL':
            display_name = 'HumanEval'
        elif dataset == 'MBPP':
            display_name = 'MBPP'
        else:
            display_name = dataset.capitalize()

        # Use | separator for all but last dataset
        separator = "|" if i < len(datasets) - 1 else ""
        dataset_parts.append(f"\\multicolumn{{{len(metrics)}}}{{c{separator}}}{{\\textbf{{{display_name}}}}}")

    dataset_row = "& " + " & ".join(dataset_parts) + " \\\\"

    # Metric row
    metric_parts = ["\\textbf{Method}"]
    for _ in datasets:
        for metric in metrics:
            metric_parts.append(metric)

    metric_row = " & ".join(metric_parts) + " \\\\"

    return dataset_row + "\n" + metric_row


def save_latex(content: str, output_file: str) -> None:
    """
    Save LaTeX content to file.

    Args:
        content: LaTeX content string
        output_file: Path to output file

    Example:
        >>> save_latex("\\begin{table}...\\end{table}", "output.tex")
    """
    with open(output_file, 'w') as f:
        f.write(content)
