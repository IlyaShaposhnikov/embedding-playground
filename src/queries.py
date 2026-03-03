"""
Query operations on embedding models: nearest neighbors and analogies.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

from gensim.models import KeyedVectors

from src.services.embedding import find_nearest_neighbors, solve_analogy
from src.presentation.formatting import (
    format_nearest_neighbors, format_analogy_results
)
from src.visualize import visualize_analogy

logger = logging.getLogger(__name__)


def nearest_neighbors(
    word: str,
    model: KeyedVectors,
    topn: int = 5,
    model_name: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """
    Find nearest neighbors of a word in the embedding space.
    Returns list of (neighbor, similarity) tuples.
    Returns empty list if word not found.
    """
    if model is None:
        print("Load a model first.")
        return []

    results = find_nearest_neighbors(word, model, topn)
    if not results:
        print(f"Word '{word}' not in vocabulary.")
        sample = list(model.key_to_index.keys())[:10]
        print(f"Sample vocabulary: {', '.join(sample)}")
        return []

    # Pretty output
    formatted_output = format_nearest_neighbors(word, results, model_name)
    print(formatted_output)

    return results


def find_analogies(
    w1: str,
    w2: str,
    w3: str,
    model: KeyedVectors,
    topn: int = 3,
    model_name: Optional[str] = None,
    visualize: bool = False,
    method: str = "pca",
    save: Optional[Path] = None,
) -> List[Tuple[str, float]]:
    """Solve word analogy: w1 - w2 = ? - w3   (vector: w1 - w2 + w3)"""
    results = solve_analogy(w1, w2, w3, model, topn=topn)

    if not results:
        if model is None:
            print("Load a model first.")
        else:
            missing = [
                word for word in (w1, w2, w3) if word not in model.key_to_index
            ]
            if missing:
                print(
                    f"Words not in vocabulary: {', '.join(missing)}"
                )
        return []

    # Pretty output
    formatted_output = format_analogy_results(w1, w2, w3, results, model_name)
    print(formatted_output)

    # Visualization with auto-saving if requested
    if visualize and model is not None and results:
        try:
            visualize_analogy(
                w1, w2, w3, results,
                model,
                model_name=model_name or "Model",
                method=method,
                save=save
            )
        except Exception as e:
            print(f"Visualization failed: {e}")
    elif visualize and not results:
        print(
            "Cannot visualize: no valid results found "
            "(words may be missing from vocabulary)."
        )

    return results
