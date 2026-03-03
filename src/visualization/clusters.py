"""
Logic for visualizing semantic clusters.
Delegates data preparation and plotting.
"""

import logging
from pathlib import Path
from typing import Optional

from gensim.models import KeyedVectors

from .data_preparation import prepare_cluster_data
from .projections import project_words
from .plotting import plot_embeddings

logger = logging.getLogger(__name__)


def visualize_word_clusters(
    seed_words: list,
    model: KeyedVectors,
    topn: int = 3,
    method: str = "pca",
    model_name: str = "Model",
    save: Optional[Path] = None,
) -> None:
    """
    Visualize semantic clusters formed by seed words
    and their nearest neighbors.
    """
    if model is None:
        print("Load a model first.")
        return

    # Prepare data using service
    words, labels, total_words = prepare_cluster_data(seed_words, model, topn)

    if words is None or labels is None:
        if total_words is not None and total_words < 2:
            print("Insufficient words for projection.")
        else:
            # Likely missing seed words
            missing = [w for w in seed_words if w not in model.key_to_index]
            if missing:
                print(f"Seed words not in vocabulary: {', '.join(missing)}")
        return

    if len(seed_words) > 6:
        print(
            f"Warning: {len(seed_words)} seed words "
            "may produce a crowded plot. "
            "Consider using 3-4 words for better readability."
        )

    print(
        f"Collected {total_words} words: {len(seed_words)} seeds "
        f"+ up to {topn} neighbors each."
    )

    # Project
    coords = project_words(model, words, method=method)
    if coords is None:
        return

    # Plot with colors
    title = (
        f"{model_name} - {method.upper()} | Semantic Clusters | "
        f"(Seeds: {', '.join(seed_words)})"
    )
    plot_embeddings(
        coords,
        words,
        labels,
        seed_words,
        title=title,
        save_path=save,
    )
