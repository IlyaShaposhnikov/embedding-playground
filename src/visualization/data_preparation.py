"""
Pure logic for preparing data for visualization.
Does not perform any plotting or printing.
"""

from typing import List, Tuple, Optional

from gensim.models import KeyedVectors

from src.data.data_extraction import get_nearest_neighbors


def prepare_cluster_data(
    seed_words: list,
    model: KeyedVectors,
    topn: int = 3,
) -> Tuple[Optional[List[str]], Optional[List[int]], Optional[int]]:
    """
    Prepare data for cluster visualization.
    Returns:
        - List of words to visualize
        - List of cluster labels for each word
        - Total number of collected words
    """
    if model is None:
        return None, None, None

    # Validate seed words
    missing = [w for w in seed_words if w not in model.key_to_index]
    if missing:
        # Could return error information, for now just return None
        return None, None, None

    # Collect words and assign cluster labels based on seed index
    word_to_cluster = {}
    cluster_words = []  # list of (word, cluster_id)

    for idx, seed in enumerate(seed_words):
        # Add seed word itself
        if seed not in word_to_cluster:
            word_to_cluster[seed] = idx
            cluster_words.append((seed, idx))

        # Fetch neighbors using the shared nearest-neighbor logic
        neighbors = get_nearest_neighbors(seed, model, topn=topn)
        if neighbors:
            for neighbor, _ in neighbors:
                if neighbor not in word_to_cluster:
                    word_to_cluster[neighbor] = idx
                    cluster_words.append((neighbor, idx))
        # else: # Optionally log that no neighbors were found for a seed

    total_words = len(cluster_words)

    if total_words < 2:
        # Insufficient words for projection
        return None, None, total_words

    # Extract words and labels
    words = [w for w, _ in cluster_words]
    labels = [c for _, c in cluster_words]

    return words, labels, total_words

# Could add similar function for analogy preparation if needed
