"""
Logic for plotting 2D projections.
Does not perform data preparation or printing messages about saving.
"""

from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


def plot_embeddings(
    coords: np.ndarray,
    words: list,
    labels: Optional[list] = None,
    seed_words: Optional[list] = None,
    title: str = "Word Embeddings Visualization",
    figsize: tuple = (12, 8),
    save_path: Optional[Path] = None,
) -> None:
    """Plot 2D projections with optional colored clusters and word labels."""
    if coords is None or len(coords) == 0:
        print("No coordinates to plot.")
        return

    plt.figure(figsize=figsize)

    if labels is not None and seed_words is not None:
        # Assign distinct colors to clusters using tab10 colormap
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for cluster_id in unique_labels:
            mask = [i for i, label in enumerate(labels) if label == cluster_id]
            plt.scatter(
                coords[mask, 0],
                coords[mask, 1],
                color=colors[cluster_id],
                label=f"Cluster: {seed_words[cluster_id]}",
                alpha=0.7,
                edgecolors="k",
                s=100
            )
    else:
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6, edgecolors="k")

    for i, word in enumerate(words):
        plt.annotate(
            word,
            (coords[i, 0], coords[i, 1]),
            fontsize=9,
            alpha=0.9,
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, linestyle="--", alpha=0.5)

    if labels is not None:
        plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_analogy(
    coords: np.ndarray,
    words: List[str],
    labels: List[int],
    w1_idx: Optional[int],
    w2_idx: Optional[int],
    w3_idx: Optional[int],
    result_indices: List[int],
    title: str,
    save_path: Optional[Path] = None,
) -> None:
    """Plot analogy with vector arrows and semantic coloring."""
    plt.figure(figsize=(14, 10))

    # Color mapping:
    # 0 (w1) and 3 (results) -> blue, 1 (w2) -> red, 2 (w3) -> green
    colors = {
        0: '#1f77b4',  # blue for w1
        1: '#d62728',  # red for w2
        2: '#2ca02c',  # green for w3
        3: '#1f77b4',  # blue for predicted results (same as w1 side)
    }

    marker_map = {
        0: 'o',  # w1 — circle
        1: 's',  # w2 — square
        2: '^',  # w3 — triangle
        3: 'D',  # results — diamond
    }

    for i, (word, label) in enumerate(zip(words, labels)):
        plt.scatter(
            coords[i, 0], coords[i, 1],
            color=colors.get(label, '#7f7f7f'),
            marker=marker_map.get(label, 'o'),
            s=300 if i in [w1_idx, w2_idx, w3_idx] + result_indices else 150,
            edgecolors='k',
            linewidths=1.5,
            alpha=0.9,
            # Higher zorder brings key points to front
            zorder=3 if i in [w1_idx, w2_idx, w3_idx] + result_indices else 2
        )
        plt.annotate(
            word,
            (coords[i, 0], coords[i, 1]),
            fontsize=(
                11
                if i in [w1_idx, w2_idx, w3_idx] + result_indices
                else 9
            ),
            alpha=0.95,
            xytext=(8, 8),
            textcoords='offset points',
            fontweight=(
                'bold'
                if i in [w1_idx, w2_idx, w3_idx] + result_indices
                else 'normal'
            ),
            zorder=4  # annotations above points
        )

    # Draw arrow from w2 to w1 representing (w1 - w2)
    if w1_idx is not None and w2_idx is not None:
        # w2 → w1
        arrow = FancyArrowPatch(
            (coords[w2_idx, 0], coords[w2_idx, 1]),
            (coords[w1_idx, 0], coords[w1_idx, 1]),
            arrowstyle='->,head_width=0.8,head_length=1.2',
            color='#d62728',
            linewidth=2.5,
            alpha=0.7,
            zorder=1  # arrows behind points
        )
        plt.gca().add_patch(arrow)
        # Place text at midpoint of arrow
        plt.text(
            (coords[w2_idx, 0] + coords[w1_idx, 0]) / 2,
            (coords[w2_idx, 1] + coords[w1_idx, 1]) / 2,
            'w1 - w2',
            fontsize=10,
            color='#d62728',
            fontweight='bold',
            ha='center',
            va='bottom'
        )

    # Draw arrow from the top result to w3 representing (? - w3)
    if w3_idx is not None and result_indices:
        # result → w3
        result_idx = result_indices[0]  # use the top predicted word
        arrow = FancyArrowPatch(
            (coords[result_idx, 0], coords[result_idx, 1]),
            (coords[w3_idx, 0], coords[w3_idx, 1]),
            arrowstyle='->,head_width=0.8,head_length=1.2',
            color='#2ca02c',
            linewidth=2.5,
            alpha=0.7,
            zorder=1
        )
        plt.gca().add_patch(arrow)
        plt.text(
            (coords[result_idx, 0] + coords[w3_idx, 0]) / 2,
            (coords[result_idx, 1] + coords[w3_idx, 1]) / 2,
            '? - w3',
            fontsize=10,
            color='#2ca02c',
            fontweight='bold',
            ha='center',
            va='bottom'
        )

    # Build custom legend
    legend_elements = [
        plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
            markersize=12, label=(
                'w1 / Results '
                f'({words[w1_idx] if w1_idx is not None else "?"})'
            )
        ),
        plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor='#d62728',
            markersize=12, label=(
                'w2 '
                f'({words[w2_idx] if w2_idx is not None else "?"})'
            )
        ),
        plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
            markersize=12, label=(
                'w3 '
                f'({words[w3_idx] if w3_idx is not None else "?"})'
            )
        ),
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=10)

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Component 1", fontsize=12)
    plt.ylabel("Component 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
