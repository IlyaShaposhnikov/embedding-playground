"""
Pure business logic for core embedding operations.
Does not perform any formatting or printing.
"""

from typing import List, Tuple

from gensim.models import KeyedVectors

from src.data.data_extraction import (
    get_nearest_neighbors, get_analogy_solution
)


def find_nearest_neighbors(
    word: str,
    model: KeyedVectors,
    topn: int = 5,
) -> List[Tuple[str, float]]:
    """
    Find nearest neighbors of a word in the embedding space.
    Returns list of (neighbor, similarity) tuples.
    Returns empty list if word not found.
    """
    if model is None:
        return []

    results = get_nearest_neighbors(word, model, topn)
    return results or []


def solve_analogy(
    w1: str,
    w2: str,
    w3: str,
    model: KeyedVectors,
    topn: int = 3,
) -> List[Tuple[str, float]]:
    """
    Solve word analogy: w1 - w2 = ? - w3   (vector: w1 - w2 + w3).
    Returns list of (candidate, similarity) tuples.
    Returns empty list if any word not found or solution fails.
    """
    if model is None:
        return []

    results = get_analogy_solution(w1, w2, w3, model, topn)
    return results or []
