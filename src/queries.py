"""
Query operations on embedding models: nearest neighbors and analogies.
"""

from typing import List, Tuple
from gensim.models import KeyedVectors


def nearest_neighbors(
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
        print("Load a model first.")
        return []

    if word not in model.key_to_index:
        print(f"Word '{word}' not in vocabulary.")
        # Show some random words from vocabulary as hint
        sample = list(model.key_to_index.keys())[:10]
        print(f"Sample vocabulary: {', '.join(sample)}")
        return []

    try:
        results = model.most_similar(positive=[word], topn=topn)

        # Pretty print
        print(f"\nNEAREST NEIGHBORS: '{word}'")
        print("─" * 60)
        print(f"{'#':>2s}  {'Word':<20s}  {'Similarity':>10s}")
        print("─" * 60)
        for i, (neighbor, sim) in enumerate(results, 1):
            print(f"{i:2d}. {neighbor:<20s}  {sim:>10.4f}")
        print("─" * 60)

        return results

    except Exception as e:
        print(f"Error during nearest neighbor search: {e}")
        return []


def find_analogies(
    w1: str,
    w2: str,
    w3: str,
    model: KeyedVectors,
    topn: int = 3,
) -> List[Tuple[str, float]]:
    """Solve word analogy: w1 - w2 = ? - w3   (vector: w1 - w2 + w3)"""
    if model is None:
        print("Load a model first.")
        return []

    # Check that all input words exist in vocabulary
    missing = [word for word in (w1, w2, w3) if word not in model.key_to_index]
    if missing:
        print(f"Words not in vocabulary: {', '.join(missing)}")
        return []

    try:
        # Vector arithmetic: w1 - w2 + w3
        results = model.most_similar(
            positive=[w1, w3], negative=[w2], topn=topn
        )

        # Pretty output
        print(f"\nANALOGY: {w1} - {w2} = ? - {w3}")
        print("─" * 60)
        print(f"{'#':>2s}  {'Solution':<20s}  {'Similarity':>10s}")
        print("─" * 60)
        for i, (candidate, sim) in enumerate(results, 1):
            print(f"{i:2d}. {candidate:<20s}  {sim:>10.4f}")
        print("─" * 60)

        return results

    except Exception as e:
        print(f"Error during analogy search: {e}")
        return []
