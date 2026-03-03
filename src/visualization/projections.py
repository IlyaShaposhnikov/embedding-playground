"""
Logic for projecting word vectors into 2D space.
Does not perform any plotting or printing.
"""

import logging
from typing import Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)

# Default visualization settings
DEFAULT_N_WORDS = 50
DEFAULT_METHOD = "pca"
RANDOM_SEED = 42  # Fixed seed for reproducible PCA/t-SNE results


def project_words(
    model: KeyedVectors,
    words: list,
    method: str = "pca",
    perplexity: int = 30,
    random_state: int = RANDOM_SEED,
) -> Optional[np.ndarray]:
    """Project word vectors into 2D space using PCA or t-SNE."""
    if not words:
        print("No words provided.")
        # Or return None and let caller handle
        return None

    # Filter words present in vocabulary
    valid_words = [w for w in words if w in model.key_to_index]
    if not valid_words:
        print("None of the provided words are in the vocabulary.")
        # Or return None
        return None

    if len(valid_words) < len(words):
        print(
            f"Skipped {len(words) - len(valid_words)} word(s) "
            "not in vocabulary."
        )

    # Get vectors
    vectors = model[valid_words]

    try:
        if method == "pca":
            if len(valid_words) < 2:
                print(
                    "PCA requires at least 2 points "
                    f"(got {len(valid_words)}). "
                    "Provide more words."
                )
                return None
            reducer = PCA(n_components=2, random_state=random_state)
        elif method == "tsne":
            if len(valid_words) < 3:
                print(
                    "t-SNE requires at least 3 points "
                    f"(got {len(valid_words)}). "
                    "Use PCA instead or provide more words."
                )
                return None
            # t-SNE perplexity should be less than number of points.
            # We cap it at (n_samples - 2) to avoid warnings/errors.
            reducer = TSNE(
                n_components=2,
                perplexity=min(perplexity, len(valid_words) - 2),
                random_state=random_state,
                init="pca",  # Initialise with PCA for faster convergence
                learning_rate="auto",  # Let sklearn set appropriate rate
            )
        else:
            print(f"Unknown method: {method}. Use 'pca' or 'tsne'.")
            # Or return None
            return None

        coords = reducer.fit_transform(vectors)
        return coords

    except Exception as e:
        print(f"Projection failed: {e}")  # Or return None
        return None
