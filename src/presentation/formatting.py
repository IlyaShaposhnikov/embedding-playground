"""
Functions to format raw data from services into user-readable strings.
Does not perform any business logic or printing.
"""

from typing import List, Tuple, Optional


def format_nearest_neighbors(
    word: str,
    results: List[Tuple[str, float]],
    model_name: Optional[str] = None,
) -> str:
    """
    Format the results of nearest neighbor search for console output.
    """
    model_label = f"{model_name}" if model_name else ""
    output_lines = [f"\n{model_label} | NEAREST NEIGHBORS: '{word}'"]
    output_lines.append("─" * 60)
    for i, (neighbor, sim) in enumerate(results, 1):
        bar = "=" * int(sim * 20)
        output_lines.append(f"{i:2d}. {neighbor:20s} | {sim:.4f} | {bar}")
    output_lines.append("─" * 60)
    return "\n".join(output_lines)


def format_analogy_results(
    w1: str,
    w2: str,
    w3: str,
    results: List[Tuple[str, float]],
    model_name: Optional[str] = None,
) -> str:
    """
    Format the results of analogy solving for console output.
    """
    model_label = f"{model_name}" if model_name else ""
    output_lines = [f"\n{model_label} | ANALOGY: {w1} - {w2} = ? - {w3}"]
    output_lines.append("─" * 60)
    output_lines.append(f"{'#':>2s}  {'Solution':<20s}  {'Similarity':>10s}")
    output_lines.append("─" * 60)
    for i, (candidate, sim) in enumerate(results, 1):
        output_lines.append(f"{i:2d}. {candidate:<20s}  {sim:>10.4f}")
    output_lines.append("─" * 60)
    return "\n".join(output_lines)
