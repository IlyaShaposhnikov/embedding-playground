"""
Logic for visualizing word analogies.
Delegates data preparation and plotting.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

from gensim.models import KeyedVectors

from .projections import project_words
from .plotting import plot_analogy

logger = logging.getLogger(__name__)


def visualize_analogy(
    w1: str,
    w2: str,
    w3: str,
    results: List[Tuple[str, float]],
    model: KeyedVectors,
    model_name: str = "Model",
    method: str = "pca",
    save: Optional[Path] = None,
) -> None:
    """
    Visualize word analogy with vector relationships.

    Color scheme:
      • w1 and predicted words: blue (left side of analogy)
      • w2: red (subtracted word)
      • w3: green (right side of analogy)

    Shows vector arrows: w2 → w1 and predicted → w3
    """
    # Collect all words to be plotted
    words = [w1, w2, w3] + [word for word, _ in results]

    valid_words = [w for w in words if w in model.key_to_index]
    if not valid_words:
        print("No valid words for visualization.")
        return

    coords = project_words(model, valid_words, method=method)
    if coords is None or len(coords) != len(valid_words):
        return

    # Assign label: 0 for w1, 1 for w2, 2 for w3, 3 for predicted results
    word_to_idx = {word: i for i, word in enumerate(valid_words)}
    labels = []
    for word in valid_words:
        if word == w1:
            labels.append(0)
        elif word == w2:
            labels.append(1)
        elif word == w3:
            labels.append(2)
        else:
            labels.append(3)

    title = (
        f"{model_name} - {method.upper()} | Analogy: {w1} - {w2} = ? - {w3}"
    )
    plot_analogy(
        coords,
        valid_words,
        labels,
        w1_idx=word_to_idx.get(w1),
        w2_idx=word_to_idx.get(w2),
        w3_idx=word_to_idx.get(w3),
        result_indices=[
            word_to_idx.get(r[0]) for r in results if r[0] in word_to_idx
        ],
        title=title,
        save_path=save,
    )
