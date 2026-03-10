"""
Logic for projecting word vectors into 2D space.
Does not perform any plotting or printing.
"""

import logging
from typing import Optional

from gensim.models import KeyedVectors
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.core.config import settings

logger = logging.getLogger(__name__)


def project_words(
    model: KeyedVectors,
    words: list,
    method: str = settings.DEFAULT_VIZ_METHOD,
    perplexity: int = settings.DEFAULT_TSNE_PERPLEXITY,
    random_state: int = settings.DEFAULT_RANDOM_STATE,
) -> Optional[np.ndarray]:
    """Project word vectors into 2D space using PCA or t-SNE."""
    if not words:
        logger.warning("No words provided.")
        # Or return None and let caller handle
        return None

    # Filter words present in vocabulary
    valid_words = [w for w in words if w in model.key_to_index]
    if not valid_words:
        logger.warning("None of the provided words are in the vocabulary.")
        # Or return None
        return None

    if len(valid_words) < len(words):
        logger.info(
            f"Skipped {len(words) - len(valid_words)} word(s) "
            "not in vocabulary."
        )

    # Get vectors
    vectors = model[valid_words]

    try:
        if method == "pca":
            if len(valid_words) < settings.DEFAULT_PCA_N_COMPONENTS:
                logger.warning(
                    "PCA requires at least 2 points "
                    f"(got {len(valid_words)}). "
                    "Provide more words."
                )
                return None
            reducer = PCA(
                n_components=settings.DEFAULT_PCA_N_COMPONENTS,
                random_state=random_state
            )
        elif method == "tsne":
            if len(valid_words) < 3:
                logger.warning(
                    "t-SNE requires at least 3 points "
                    f"(got {len(valid_words)}). "
                    "Use PCA instead or provide more words."
                )
                return None
            # t-SNE perplexity should be less than number of points.
            # We cap it at (n_samples - 2) to avoid warnings/errors.
            reducer = TSNE(
                n_components=settings.DEFAULT_TSNE_N_COMPONENTS,
                perplexity=min(perplexity, len(valid_words) - 2),
                random_state=random_state,
                init="pca",  # Initialise with PCA for faster convergence
                learning_rate="auto",  # Let sklearn set appropriate rate
            )
        else:
            logger.error(f"Unknown method: {method}. Use 'pca' or 'tsne'.")
            # Or return None
            return None

        coords = reducer.fit_transform(vectors)
        return coords

    except Exception as e:
        logger.exception(f"Projection failed: {e}")  # Or return None
        return None
