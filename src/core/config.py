"""
Centralized configuration
"""

from pathlib import Path


class Settings:
    """Application-wide settings."""

    # --- Paths ---
    DATA_DIR = Path("data")
    VIZ_DIR = DATA_DIR / "visualizations"
    LOGS_DIR = Path("logs")

    # --- Word2Vec Configuration ---
    class Word2Vec:
        # Primary and mirror URLs for the GoogleNews vectors
        # The mmihaltz GitHub repo provides an alternative to Google Drive
        URL = "https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"  # noqa: E501
        URL_MIRROR = "https://github.com/mmihaltz/word2vec-GoogleNews-vectors/raw/master/GoogleNews-vectors-negative300.bin.gz"  # noqa: E501
        # Compressed file name
        GZ_NAME = "GoogleNews-vectors-negative300.bin.gz"
        # Name of the uncompressed file after extraction
        BIN_NAME = "GoogleNews-vectors-negative300.bin"
        # Expected size of the compressed file (in bytes)
        GZ_SIZE = 1_647_046_227  # ~1.53 GB
        # Expected size of the uncompressed .bin file (in bytes)
        BIN_SIZE = 3_644_258_522  # ~3.39 GB
        # Disk space buffer
        BUFFER = 500_000_000  # 500 MB buffer

    # --- GloVe Configuration ---
    class GloVe:
        # URL for the GloVe zip archive containing multiple versions
        URL = "https://nlp.stanford.edu/data/glove.6B.zip"
        # Expected size of the zip archive (in bytes)
        ZIP_SIZE = 862_182_613  # ~822 MB
        # Expected sizes for individual .txt files
        # (uncompressed) based on dimensions
        TXT_SIZES = {
            "6B.50d": 171_350_079,    # ~163 MB
            "6B.100d": 347_116_733,   # ~331 MB
            "6B.200d": 693_432_828,   # ~661 MB
            "6B.300d": 1_037_962_819  # ~989 MB
        }
        # Default version to use if none specified
        DEFAULT_VERSION = "6B.100d"
        # Name pattern for GloVe .txt files
        TXT_PATTERN = "glove.{version}.txt"
        # Name of the zip archive
        ZIP_NAME = "glove.6B.zip"
        # Disk space buffer
        BUFFER = 200_000_000  # 200 MB buffer

    # --- Analogy Test Set Configuration ---
    class AnalogyTestSet:
        URL = "http://download.tensorflow.org/data/questions-words.txt"
        TXT_NAME = "questions-words.txt"
        # Minimum expected file size in bytes
        MIN_SIZE = 500_000
        # Classification of sections based on the categories
        # defined in the original paper.
        # These sets are used to separate semantic and syntactic accuracy.
        SEMANTIC_SECTIONS = {
            'capital-common-countries', 'currency',
            'city-in-state', 'family', 'gram6-nationality-adjective'
        }
        SYNTACTIC_SECTIONS = {
            'gram1-adjective-to-adverb', 'gram2-opposite', 'gram3-comparative',
            'gram4-superlative', 'gram5-present-participle',
            'gram7-past-tense', 'gram8-plural', 'gram9-plural-verbs'
        }

    # --- Download Configuration ---
    class Download:
        # Threshold for file size mismatch warnings (in percent)
        THRESHOLD = 5.0

    # --- Demo Configuration ---
    class Demo:
        NEIGHBORS = ["king", "france", "computer"]
        ANALOGIES = [
            ("king", "man", "woman"),
            ("france", "paris", "london"),
            ("moscow", "russia", "tokyo"),
        ]

    # --- Application Limits and Defaults ---
    MAX_TOPN = 50
    MAX_NEIGHBORS_PER_WORD_VC = 20
    DEFAULT_TOPN_NN = 5
    DEFAULT_TOPN_ANA = 3
    DEFAULT_TOPN_VC = 3
    DEFAULT_TOPN_EVAL = 1

    # --- Visualization Settings ---
    DEFAULT_VIZ_METHOD = "pca"
    DEFAULT_PCA_N_COMPONENTS = 2
    DEFAULT_TSNE_N_COMPONENTS = 2
    DEFAULT_TSNE_PERPLEXITY = 30
    # For reproducible PCA/t-SNE results
    DEFAULT_RANDOM_STATE = 42
    # For general visualization functions
    DEFAULT_N_WORDS_VISUALIZE = 50

    # --- Model Loading ---
    # Default directory relative to project root
    MODELS_DIR = DATA_DIR


# Instantiate the settings object for easy access
settings = Settings()
