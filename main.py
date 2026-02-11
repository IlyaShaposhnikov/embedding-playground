"""
Entry point for embedding-playground.
Currently tests the Word2Vec download mechanism.
"""

from src.download import download_word2vec_model


def main():
    try:
        print("=" * 60)
        print("Embedding Playground — Word2Vec Download Test")
        print("=" * 60)

        model_path = download_word2vec_model()
        if model_path:
            print(f"\nModel ready at: {model_path}")
            print(
                "You can now load it with "
                "gensim.models.KeyedVectors.load_word2vec_format()"
            )
        else:
            print("\nModel download failed.")
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nUnexpected error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
