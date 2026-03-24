"""CLI entrypoint for the n-gram phonetic language model pipeline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

try:  # pragma: no cover - allow running as script or module
    from .data_pipeline import read_utterance_file, split_corpus
    from .models import NGramLanguageModel
    from .persistence import load_model_bundle, save_model_bundle
except ImportError:  # pragma: no cover
    from data_pipeline import read_utterance_file, split_corpus
    from models import NGramLanguageModel
    from persistence import load_model_bundle, save_model_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ngram_lm",
        description="Corpus preparation and n-gram language modelling pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser(
        "split",
        help="Create deterministic train/dev splits from transformed CHILDES data.",
    )
    split_parser.add_argument("--input", required=True, help="Directory containing transformed .txt files.")
    split_parser.add_argument("--train-out", required=True, help="Destination path for training corpus.")
    split_parser.add_argument("--dev-out", required=True, help="Destination path for dev corpus.")
    split_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of utterances allocated to the training set (default: 0.8).",
    )
    split_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for deterministic shuffling (default: system entropy).",
    )
    split_parser.set_defaults(func=handle_split)

    train_parser = subparsers.add_parser(
        "train",
        help="Train n-gram language models and persist JSON artifacts.",
    )
    train_parser.add_argument("--input", required=True, help="Path to training corpus (one utterance per line).")
    train_parser.add_argument("--output", required=True, help="Destination JSON file for trained models.")
    train_parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        required=True,
        help="List of n-gram orders to train (e.g., 1 2 3).",
    )
    train_parser.add_argument(
        "--disable-unk",
        action="store_true",
        help="Disable unknown-token back-off handling during training.",
    )
    train_parser.set_defaults(func=handle_train)

    eval_parser = subparsers.add_parser(
        "eval",
        help="Compute perplexity for a saved model against a dataset.",
    )
    eval_parser.add_argument("--model-file", required=True, help="JSON file produced by the train command.")
    eval_parser.add_argument("--order", type=int, required=True, help="Order of the model to evaluate.")
    eval_parser.add_argument("--data", required=True, help="Dataset to score (one utterance per line).")
    eval_parser.add_argument(
        "--smoothing",
        choices=["none", "laplace"],
        default="none",
        help="Smoothing strategy to apply during evaluation (default: none).",
    )
    eval_parser.set_defaults(func=handle_eval)

    return parser


def handle_split(args: argparse.Namespace) -> None:
    transformed_dir = Path(args.input)
    train_out = Path(args.train_out)
    dev_out = Path(args.dev_out)

    summary = split_corpus(
        transformed_dir=transformed_dir,
        train_out=train_out,
        dev_out=dev_out,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    metadata = {
        "transformed_dir": str(transformed_dir.resolve()),
        **summary,
    }

    print("Split complete. Metadata:")
    print(json.dumps(metadata, indent=2))


def handle_train(args: argparse.Namespace) -> None:
    corpus_path = Path(args.input)
    output_path = Path(args.output)
    utterances = read_utterance_file(corpus_path)

    unique_orders = sorted({order for order in args.orders if order >= 1})
    if not unique_orders:
        raise ValueError("At least one positive n-gram order must be specified.")

    models = {}
    for order in unique_orders:
        model = NGramLanguageModel(order, use_unknown_token=not args.disable_unk)
        model.fit(utterances)
        models[order] = model

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "orders": unique_orders,
        "source_corpus": str(corpus_path.resolve()),
        "utterance_count": len(utterances),
    }

    save_model_bundle(models, output_path, metadata=metadata)

    print(
        f"Trained {len(models)} model(s) (orders: {unique_orders}) "
        f"and saved artifacts to {output_path}"
    )


def handle_eval(args: argparse.Namespace) -> None:
    model_file = Path(args.model_file)
    dataset_path = Path(args.data)

    models, metadata = load_model_bundle(model_file)
    if args.order not in models:
        raise ValueError(
            f"Requested order {args.order} not present in {model_file}. Available: {sorted(models.keys())}"
        )

    utterances = read_utterance_file(dataset_path)
    model = models[args.order]
    perplexity = model.perplexity(utterances, smoothing=args.smoothing)

    print(
        json.dumps(
            {
                "model_file": str(model_file.resolve()),
                "order": args.order,
                "smoothing": args.smoothing,
                "perplexity": perplexity,
                "dataset": str(dataset_path.resolve()),
                "metadata": metadata,
            },
            indent=2,
        )
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main()
