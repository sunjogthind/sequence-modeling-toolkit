"""Data ingestion and splitting utilities for the language model pipeline."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class CorpusSummary:
    """Lightweight statistics about a phonetic corpus."""

    utterance_count: int
    token_count: int
    vocabulary_size: int
    average_utterance_length: float

    def as_dict(self) -> dict:
        return {
            "utterance_count": self.utterance_count,
            "token_count": self.token_count,
            "vocabulary_size": self.vocabulary_size,
            "average_utterance_length": self.average_utterance_length,
        }


def collect_utterances(root: Path) -> List[List[str]]:
    """Traverse ``root`` and aggregate phonetic utterances.

    The function expects a directory containing text files where each non-empty
    line represents a whitespace-delimited sequence of phonetic tokens.
    Duplicate utterances are automatically eliminated to improve corpus
    diversity.
    """

    if not root.exists():
        raise FileNotFoundError(f"Input directory {root} does not exist")

    sequences: List[List[str]] = []
    seen: set[tuple[str, ...]] = set()

    for path in sorted(root.rglob("*.txt")):
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            tokens = raw_line.strip().split()
            if not tokens:
                continue
            key = tuple(tokens)
            if key in seen:
                continue
            seen.add(key)
            sequences.append(tokens)

    if not sequences:
        raise ValueError(f"No utterances discovered under {root}")

    return sequences


def write_utterances(utterances: Sequence[Sequence[str]], destination: Path) -> None:
    """Persist utterances to disk as space-delimited lines."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for sequence in utterances:
            handle.write(" ".join(sequence).strip())
            handle.write("\n")


def split_corpus(
    transformed_dir: Path,
    train_out: Path,
    dev_out: Path,
    train_ratio: float = 0.8,
    seed: int | None = None,
) -> dict:
    """Create train/dev splits from ``transformed_dir`` and save artifacts."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    utterances = collect_utterances(transformed_dir)
    random_generator = random.Random(seed)
    random_generator.shuffle(utterances)

    split_index = int(len(utterances) * train_ratio)
    train_sequences = utterances[:split_index]
    dev_sequences = utterances[split_index:]

    write_utterances(train_sequences, train_out)
    write_utterances(dev_sequences, dev_out)

    return {
        "train": summarise_corpus(train_sequences).as_dict(),
        "dev": summarise_corpus(dev_sequences).as_dict(),
        "seed": seed,
        "train_ratio": train_ratio,
    }


def read_utterance_file(path: Path) -> List[List[str]]:
    """Load a newline-delimited utterance file into memory."""

    if not path.exists():
        raise FileNotFoundError(path)

    utterances: List[List[str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        tokens = raw_line.strip().split()
        if tokens:
            utterances.append(tokens)

    if not utterances:
        raise ValueError(f"No utterances found in {path}")

    return utterances


def summarise_corpus(utterances: Iterable[Sequence[str]]) -> CorpusSummary:
    """Compute descriptive statistics for the provided utterances."""

    utterances = list(utterances)
    if not utterances:
        return CorpusSummary(0, 0, 0, 0.0)

    vocab: set[str] = set()
    token_total = 0
    for sequence in utterances:
        vocab.update(sequence)
        token_total += len(sequence)

    average_length = token_total / len(utterances)

    return CorpusSummary(
        utterance_count=len(utterances),
        token_count=token_total,
        vocabulary_size=len(vocab),
        average_utterance_length=average_length,
    )
