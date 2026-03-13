"""
Model Manager

This module provides a central class for managing the lifecycle
of embedding models (Word2Vec, GloVe), including lazy loading
and caching.
"""
import logging
from pathlib import Path
from typing import Dict, Optional

from gensim.models import KeyedVectors

from src.core.config import settings
from src.models import load_word2vec_model, load_glove_model

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages loading, caching, and providing access to embedding models.
    Uses lazy loading: models are loaded only when first requested.
    """

    def __init__(self):
        self._w2v_model: Optional[KeyedVectors] = None
        self._glove_model: Optional[KeyedVectors] = None

        # Paths for model files
        # self._w2v_bin_path: Optional[Path] = None
        # self._glove_txt_path: Optional[Path] = None

        # Track loading attempts to avoid repeated failures
        self._w2v_load_attempted = False
        self._glove_load_attempted = False

    def get_word2vec_model(self) -> Optional[KeyedVectors]:
        """
        Get the Word2Vec model, loading it if necessary.

        Returns:
            Loaded KeyedVectors model or None if loading fails.
        """
        if self._w2v_model is not None:
            logger.debug("Returning cached Word2Vec model.")
            return self._w2v_model

        if self._w2v_load_attempted:
            logger.info(
                "Word2Vec model loading was attempted before and failed "
                "or returned None. Returning None."
            )
            return None

        logger.info("Loading Word2Vec model (lazy loading).")
        model_path = Path(settings.MODELS_DIR) / settings.Word2Vec.BIN_NAME
        self._w2v_model = load_word2vec_model(
            bin_path=str(model_path),
            data_dir=str(settings.MODELS_DIR),
            use_cached=True,
            force_reload=False
        )
        self._w2v_load_attempted = True

        if self._w2v_model is not None:
            logger.info("Word2Vec model loaded and cached successfully.")
        else:
            logger.warning("Word2Vec model failed to load or was not found.")

        return self._w2v_model

    def get_glove_model(
            self, version: str = settings.GloVe.DEFAULT_VERSION
    ) -> Optional[KeyedVectors]:
        """
        Get the GloVe model, loading it if necessary.

        Args:
            version: The GloVe vector dimension/version to load
            (e.g., '6B.100d').

        Returns:
            Loaded KeyedVectors model or None if loading fails.
        """
        if self._glove_model is not None:
            logger.debug("Returning cached GloVe model.")
            return self._glove_model

        if self._glove_load_attempted:
            logger.info(
                "GloVe model loading was attempted before and "
                "failed or returned None. Returning None."
            )
            return None

        logger.info(f"Loading GloVe model (version: {version}, lazy loading).")
        txt_filename = settings.GloVe.TXT_PATTERN.format(version=version)
        model_path = Path(settings.MODELS_DIR) / txt_filename
        self._glove_model = load_glove_model(
            txt_path=str(model_path),
            version=version,
            data_dir=str(settings.MODELS_DIR),
            use_cached=True,
            force_reload=False
        )
        self._glove_load_attempted = True

        if self._glove_model is not None:
            logger.info("GloVe model loaded and cached successfully.")
        else:
            logger.warning("GloVe model failed to load or was not found.")

        return self._glove_model

    def clear_cache(self) -> None:
        """
        Clears the cached models from memory.
        Useful for testing or forcing a reload.
        """
        logger.info("Clearing model cache.")
        self._w2v_model = None
        self._glove_model = None
        self._w2v_load_attempted = False
        self._glove_load_attempted = False

    def get_available_models(self) -> Dict[str, bool]:
        """Check model availability by filesystem presence (no loading)."""
        w2v_path = Path(settings.MODELS_DIR) / settings.Word2Vec.BIN_NAME
        glove_path = (
            Path(settings.MODELS_DIR)
            / settings.GloVe.TXT_PATTERN.format(
                version=settings.GloVe.DEFAULT_VERSION
            )
        )

        return {
            "word2vec": w2v_path.exists(),
            "glove": glove_path.exists(),
        }
