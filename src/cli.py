"""
Interactive terminal interface.
Allows users to query nearest neighbors and word analogies in real time.
"""
from datetime import datetime
import logging
from pathlib import Path
import re
from typing import Optional

from gensim.models import KeyedVectors

from src.core.config import settings
from src.core.model_manager import ModelManager
from src.download import download_analogy_test_set
from src.evaluate import evaluate_model
from src.models import model_info
from src.queries import find_analogies, nearest_neighbors
from src.visualize import visualize_word_clusters

logger = logging.getLogger(__name__)


def interactive_shell(
    model_manager: ModelManager
) -> None:
    """
    Run interactive command-line interface.
    Models are loaded lazily via model_manager on first use.
    """
    current_model_instance: Optional[KeyedVectors] = None
    model_name = "None"
    # Track preference without loading
    preferred_model: Optional[str] = None

    # Check availability without loading
    available = model_manager.get_available_models()
    if available["word2vec"]:
        preferred_model = "word2vec"
        model_name = "Word2Vec (GoogleNews)"
    elif available["glove"]:
        preferred_model = "glove"
        model_name = "GloVe (6B.100d)"

    if preferred_model is None:
        print("\nNO MODELS LOADED!")
        print("To use this tool, ensure models are downloaded")
        print("\nType 'exit' to quit.\n")
    else:
        print(f"\nPreferred model: {model_name} (loads on first use)")
        print(
            "Type 'demo' to run demonstration queries, "
            "'help' for commands, 'exit' to quit.\n"
        )

    def _ensure_model_loaded() -> bool:
        """
        Load preferred model if not already loaded.
        Returns True if model is ready, False otherwise.
        """
        nonlocal current_model_instance, model_name

        if current_model_instance is not None:
            return True

        if preferred_model == "word2vec":
            current_model_instance = model_manager.get_word2vec_model()
            if current_model_instance:
                model_name = "Word2Vec (GoogleNews)"
                return True
        elif preferred_model == "glove":
            current_model_instance = model_manager.get_glove_model()
            if current_model_instance:
                model_name = "GloVe (6B.100d)"
                return True

        print("No model available. Please download models first.")
        return False

    while True:
        try:
            cmd = input("\n>>> ").strip().lower()

            if cmd in ("exit", "quit"):
                print("Goodbye!")
                break

            elif cmd == "help":
                _show_help()

            elif cmd == "demo":
                _run_demo(model_manager)

            elif cmd == "model":
                _show_model_status(current_model_instance, model_name)

            elif cmd.startswith("use "):
                # Switch model: use word2vec | use glove
                _, target = cmd.split(maxsplit=1)

                available = model_manager.get_available_models()
                available_models = [k for k, v in available.items() if v]

                if target == "word2vec" and available["word2vec"]:
                    # Load on demand when switching
                    model = model_manager.get_word2vec_model()
                    if model:
                        preferred_model = "word2vec"
                        current_model_instance = model
                        model_name = "Word2Vec (GoogleNews)"
                        print(f"Switched to {model_name}")
                    else:
                        print(f"Failed to load {target}.")
                elif target == "glove" and available["glove"]:
                    # Load on demand when switching
                    model = model_manager.get_glove_model()
                    if model:
                        preferred_model = "glove"
                        current_model_instance = model
                        model_name = "GloVe (6B.100d)"
                        print(f"Switched to {model_name}")
                    else:
                        print(f"Failed to load {target}.")
                else:
                    print(f"Model '{target}' not available.")
                    if available_models:
                        print(f"Available: {', '.join(available_models)}")
                    else:
                        print("No models found. Download first.")

            elif cmd.startswith("nn "):
                # Nearest neighbors: nn king [5]
                parts = cmd.split()
                if len(parts) < 2:
                    print("Usage: nn <word> [topn]")
                    continue
                word = parts[1]
                try:
                    topn = (
                        int(parts[2]) if len(parts) > 2
                        else settings.DEFAULT_TOPN_NN
                    )
                    if topn < 1:
                        print("topn must be at least 1.")
                        continue
                    if topn > settings.MAX_TOPN:
                        print(
                            "Warning: topn capped at "
                            f"{settings.MAX_TOPN} (requested {topn})"
                        )
                        topn = settings.MAX_TOPN
                except ValueError:
                    print(
                        f"Invalid number: '{parts[2]}'. "
                        "Please use an integer (e.g., 5)."
                    )
                    continue
                if not _ensure_model_loaded():
                    continue
                else:
                    nearest_neighbors(
                        word, current_model_instance,
                        topn=topn, model_name=model_name
                    )

            elif cmd.startswith("ana "):
                # Analogy: ana king man woman [topn] [-v] [pca|tsne]
                # Parsing order: -v flag → w1,w2,w3 → method (end) → topn (end)
                parts = cmd.split()
                if len(parts) < 4:
                    print("Usage: ana <w1> <w2> <w3> [topn] [-v] [pca|tsne]")
                    continue

                # Extract visualization flag (-v) from anywhere in command
                visualize = "-v" in parts
                if visualize:
                    parts.remove("-v")

                # Re-validate after removing -v (command may become invalid)
                if len(parts) < 4:
                    print("Usage: ana <w1> <w2> <w3> [topn] [-v] [pca|tsne]")
                    continue
                w1, w2, w3 = parts[1:4]

                # Prepare remaining arguments for method/topn parsing
                remaining_args = parts[4:]

                # Parse visualization method from the end (pca|tsne)
                method, remaining_args = _parse_method(
                    remaining_args, settings.DEFAULT_VIZ_METHOD
                )
                # Parse topn from what remains (validates range & format)
                topn = _parse_topn(
                    remaining_args,
                    settings.DEFAULT_TOPN_ANA,
                    settings.MAX_TOPN,
                    "ana"
                )
                if topn is None:
                    # Invalid topn value — skip command
                    continue

                # Execute analogy query if model is loaded
                if not _ensure_model_loaded():
                    continue
                else:
                    # Generate save path only if visualization is requested
                    save_path = None
                    if visualize:
                        save_path = _generate_viz_save_path(
                            [w1, w2, w3], model_name, method, topn, "analogy"
                        )

                    find_analogies(
                        w1, w2, w3, current_model_instance, topn=topn,
                        model_name=model_name,
                        visualize=visualize,
                        method=method,
                        save=save_path
                    )

            elif cmd.startswith("vc "):
                # Usage: vc <word1> [word2 ...] [topn] [pca|tsne]
                # Parsing order: method (end) → topn (end) → remaining = words
                parts = cmd.split()
                if len(parts) < 2:
                    print("Usage: vc <word1> [word2 ...] [topn] [pca|tsne]")
                    continue

                # Prepare args for word parsing (exclude 'vc')
                args = parts[1:]

                # Parse visualization method from the end (pca|tsne)
                method, args = _parse_method(args, settings.DEFAULT_VIZ_METHOD)

                # Check if last argument is a number (potential topn)
                # If valid, use it as topn and remove from args
                # If invalid, treat it as a seed word and use default topn
                if args and args[-1].isdigit():
                    potential_topn_str = args[-1]
                    validated_topn = _parse_topn(
                        [potential_topn_str],
                        settings.DEFAULT_TOPN_VC,
                        settings.MAX_NEIGHBORS_PER_WORD_VC,
                        "vc"
                    )
                    if validated_topn is not None:
                        # Valid topn found — exclude it from words list
                        topn = validated_topn
                        words = args[:-1]
                    else:
                        # Invalid topn (e.g., out of range) — treat as word
                        topn = settings.DEFAULT_TOPN_VC
                        words = args
                else:
                    # No numeric argument at the end — use default topn
                    topn = settings.DEFAULT_TOPN_VC
                    words = args

                # Validate that at least one seed word is provided
                if not words:
                    print("Error: at least one seed word required.")
                    continue

                # Execute visualization if model is loaded
                if not _ensure_model_loaded():
                    continue
                else:
                    # Generate save path for visualization
                    save_path = _generate_viz_save_path(
                        words, model_name, method, topn, "clust"
                    )

                    print(
                        "Generating cluster visualization "
                        f"for seeds: {', '.join(words)} "
                        f"({topn} neighbors each, {method.upper()})..."
                    )
                    visualize_word_clusters(
                        words,
                        current_model_instance,
                        topn=topn,
                        method=method,
                        model_name=model_name,
                        save=save_path,
                    )
                    print(f"Saved to: {save_path.as_posix()}")

            elif cmd == "eval":
                if not _ensure_model_loaded():
                    continue
                else:
                    test_file = download_analogy_test_set()
                    if test_file:
                        evaluate_model(
                            current_model_instance, test_file, model_name
                        )

            elif cmd == "":
                continue

            else:
                print(f"Unknown command: {cmd}. Type 'help'.")

        except KeyboardInterrupt:
            print("\nUse 'exit' to quit.")
        except ValueError as e:
            print(f"Input error: {e}")
        except (KeyError, IndexError) as e:
            print(f"Model error: {e}")
        except Exception as e:
            print(f"Unexpected error ({type(e).__name__}): {e}")


def _parse_topn(
    args: list[str],
    default: int,
    max_limit: int,
    cmd_name: str
) -> Optional[int]:
    """
    Parses and validates the 'topn' parameter from command arguments.

    Args:
        args: List of command arguments.
        default: Default value if topn is not provided.
        max_limit: Maximum allowed value.
        cmd_name: Name of the command for error messages.

    Returns:
        Parsed and validated topn value, or None if invalid.
    """
    topn_str = args[-1] if args and args[-1].isdigit() else None
    if topn_str:
        try:
            topn = int(topn_str)
            if topn < 1:
                print(f"{cmd_name}: topn must be at least 1.")
                return None
            if topn > max_limit:
                print(
                    f"{cmd_name}: Warning: topn capped at "
                    f"{max_limit} (requested {topn})."
                )
                return max_limit
            return topn
        except ValueError:
            print(f"{cmd_name}: Invalid number: '{topn_str}'.")
            return None
    return default


def _parse_method(
        args: list[str], default_method: str
) -> tuple[str, list[str]]:
    """
    Parses the visualization method ('pca' or 'tsne') from arguments.

    Args:
        args: List of command arguments (potentially modified).
        default_method: Default method if none found.

    Returns:
        Tuple of (parsed_method, remaining_args).
    """
    method = default_method
    if args and args[-1].lower() in ("pca", "tsne"):
        method = args[-1].lower()
        # Remove the method from args
        args = args[:-1]
    return method, args


def _generate_viz_save_path(
    base_name_parts: list[str],
    model_name: str,
    method: str,
    topn: int,
    viz_type: str  # e.g., "analogy", "clust"
) -> Path:
    """
    Generates a standardized path for saving visualization plots.

    Args:
        base_name_parts: List of words or components forming the base name.
        model_name: Name of the active model.
        method: Visualization method ('pca', 'tsne').
        topn: Number of neighbors.
        viz_type: Type of visualization ("analogy", "clust").

    Returns:
        Path object for the save location.
    """
    Path(settings.VIZ_DIR).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sanitize model name for filename
    safe_model_name = re.sub(r"[^\w\-]", "_", model_name).strip("_")
    safe_model_name = re.sub(r"_+", "_", safe_model_name)

    # Create safe filename from base name parts (limit length)
    base_str = "_".join(base_name_parts)[:40]
    filename = (
        f"{viz_type}_{safe_model_name}_{method}_"
        f"{base_str}_top{topn}_{timestamp}.png"
    )
    return Path(settings.VIZ_DIR) / filename


def _show_help() -> None:
    """Display available commands."""
    help_text = """
COMMANDS:
  use <model>           Switch model: 'word2vec' or 'glove'
  nn <word> [topn]      Nearest neighbors (default topn=5)
  ana <w1> <w2> <w3> [topn] [-v] [m]
                        Word analogy (default topn=3) | w1 - w2 = ? - w3
                        • -v   : visualize results in 2D space (pca method)
                        • [m]  : method 'pca' or 'tsne' (default pca)
                        • Automatically saved to data/visualizations/
  vc <w1> [w2 ...] [n] [m]
                        Visualize semantic clusters:
                        • <w1> : seed words (min 1)
                        • [n]  : neighbors per seed (default 3, max 20)
                        • [m]  : method 'pca' or 'tsne' (default pca)
                        • Automatically saved to data/visualizations/
  demo                  Run full demonstration
                        • nearest neighbors, solve analogies, semantic clusters
  model                 Show current model info
  eval                  Evaluate current model on Google Analogy Test Set
  help                  Show this help
  exit / quit           Exit program
"""
    print(help_text)


def _show_model_status(
    model: Optional[KeyedVectors],
    name: str
) -> None:
    """Display current model status."""
    if model is None:
        print("No model loaded.")
    else:
        model_info(model, name)


def _run_demo(
        model_manager: ModelManager
) -> None:
    """Run demonstration queries for available models."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Nearest Neighbors, Analogies & Clusters")
    print("=" * 60)

    # Demo intentionally loads both models to showcase functionality
    w2v_model = model_manager.get_word2vec_model()
    glove_model = model_manager.get_glove_model()

    # Word2Vec demo
    if w2v_model:
        print("\n[Word2Vec Demo]")
        print("-" * 60)
        model_info(w2v_model, "Word2Vec (GoogleNews)")

        for word in settings.Demo.NEIGHBORS:
            nearest_neighbors(
                word, w2v_model, topn=settings.DEFAULT_TOPN_NN,
                model_name="Word2Vec (GoogleNews)"
            )

        print("\nWord2Vec Analogies")
        for w1, w2, w3 in settings.Demo.ANALOGIES:
            find_analogies(
                w1, w2, w3, w2v_model, topn=settings.DEFAULT_TOPN_ANA,
                model_name="Word2Vec (GoogleNews)"
            )

        print("\nWord2Vec Cluster Visualization")
        print(f"Seeds: {', '.join(settings.Demo.NEIGHBORS)}")
        visualize_word_clusters(
            settings.Demo.NEIGHBORS,
            w2v_model,
            topn=settings.DEFAULT_TOPN_VC,
            method=settings.DEFAULT_VIZ_METHOD,
            model_name="Word2Vec (GoogleNews)",
            save=None,
        )
    else:
        print("\nWord2Vec model not available for demo.")

    # GloVe demo
    if glove_model:
        print("\n[GloVe Demo]")
        print("-" * 60)
        model_info(glove_model, "GloVe (6B.100d)")

        for word in settings.Demo.NEIGHBORS:
            nearest_neighbors(
                word, glove_model, topn=settings.DEFAULT_TOPN_NN,
                model_name="GloVe (6B.100d)"
            )

        print("\nGloVe Analogies")
        for w1, w2, w3 in settings.Demo.ANALOGIES:
            find_analogies(
                w1, w2, w3, glove_model, topn=settings.DEFAULT_TOPN_ANA,
                model_name="GloVe (6B.100d)"
            )

        print("\nGloVe Cluster Visualization")
        print(f"Seeds: {', '.join(settings.Demo.NEIGHBORS)}")
        visualize_word_clusters(
            settings.Demo.NEIGHBORS,
            glove_model,
            topn=settings.DEFAULT_TOPN_VC,
            method=settings.DEFAULT_VIZ_METHOD,
            model_name="GloVe (6B.100d)",
            save=None,
        )
    else:
        print("\nGloVe model not available for demo.")

    print("\n" + "=" * 60)
    print("Demo completed.")
    print("=" * 60)
