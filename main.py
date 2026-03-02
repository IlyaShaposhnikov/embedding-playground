import sys

from src.cli import interactive_shell
from src.download import download_word2vec_model, download_glove_model
from src.models import load_word2vec_model, load_glove_model


def main():
    print("=" * 60)
    print("Embedding Visualizer — Interactive Shell")
    print("=" * 60)
    print("Models will be downloaded on first use if missing.\n")

    # Preload models (download if missing)
    print("[1] Preparing Word2Vec model...")
    w2v_path = download_word2vec_model()
    w2v_model = (
        load_word2vec_model(w2v_path, use_cached=True) if w2v_path else None
    )

    print("\n[2] Preparing GloVe model (6B.100d)...")
    glove_path = download_glove_model(version="6B.100d")
    glove_model = (
        load_glove_model(glove_path, use_cached=True) if glove_path else None
    )

    # Start interactive shell
    print("\n" + "=" * 60)
    print("Starting interactive shell...")
    print("=" * 60)

    interactive_shell(w2v_model, glove_model)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {type(e).__name__}: {e}")
        sys.exit(1)
