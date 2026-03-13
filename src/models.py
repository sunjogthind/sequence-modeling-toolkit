"""Core n-gram language modelling primitives."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence, Tuple

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"


@dataclass(frozen=True)
class ProbabilityResult:
    """Container for probability calculations."""

    ngram: Tuple[str, ...]
    probability: float
    count: int
    context_count: int


class NGramLanguageModel:
    """Count-based n-gram language model with optional Laplace smoothing."""

    def __init__(self, order: int, *, use_unknown_token: bool = True) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.use_unknown_token = use_unknown_token
        self.ngram_counts: Counter[Tuple[str, ...]] = Counter()
        self.context_counts: Counter[Tuple[str, ...]] = Counter()
        self.vocabulary: set[str] = set()
        self.total_observations: int = 0

    def fit(self, utterances: Iterable[Sequence[str]]) -> None:
        """Populate the model with counts derived from ``utterances``."""

        sequences = [list(sequence) for sequence in utterances if sequence]
        if not sequences:
            raise ValueError("No utterances provided to fit()")

        self.ngram_counts.clear()
        self.context_counts.clear()
        self.vocabulary = set()
        self.total_observations = 0

        for sequence in sequences:
            self.vocabulary.update(sequence)

        self.vocabulary.add(END_TOKEN)
        if self.use_unknown_token:
            self.vocabulary.add(UNKNOWN_TOKEN)

        for sequence in sequences:
            padded = [START_TOKEN] * (self.order - 1) + sequence + [END_TOKEN]
            for ngram in self._generate_ngrams(padded):
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                if context:
                    self.context_counts[context] += 1
                self.total_observations += 1

    def perplexity(
        self,
        utterances: Iterable[Sequence[str]],
        *,
        smoothing: str | None = None,
    ) -> float:
        """Compute perplexity for ``utterances`` under the model."""

        smoothing = (smoothing or "none").lower()
        self._validate_smoothing(smoothing)

        if self.total_observations == 0:
            raise ValueError("Model has not been trained yet")

        total_log_probability = 0.0
        token_count = 0
        for sequence in utterances:
            if not sequence:
                continue
            normalized = [self._map_token(token) for token in sequence]
            padded = [START_TOKEN] * (self.order - 1) + normalized + [END_TOKEN]
            for ngram in self._generate_ngrams(padded):
                probability = self._probability(ngram, smoothing)
                if probability <= 0.0:
                    return math.inf
                total_log_probability += math.log(probability)
                token_count += 1

        if token_count == 0:
            raise ValueError("No tokens available to evaluate perplexity")

        return math.exp(-total_log_probability / token_count)

    def probability_details(
        self,
        ngram: Sequence[str],
        *,
        smoothing: str | None = None,
    ) -> ProbabilityResult:
        """Return detailed probability bookkeeping for ``ngram``."""

        smoothing = (smoothing or "none").lower()
        self._validate_smoothing(smoothing)

        if self.total_observations == 0:
            raise ValueError("Model has not been trained yet")

        ngram_tuple = tuple(ngram)
        if len(ngram_tuple) != self.order:
            raise ValueError("ngram length does not match model order")

        probability = self._probability(ngram_tuple, smoothing)
        context = ngram_tuple[:-1]

        if context:
            context_count = self.context_counts.get(context, 0)
        else:
            context_count = self.total_observations

        return ProbabilityResult(
            ngram=ngram_tuple,
            probability=probability,
            count=self.ngram_counts.get(ngram_tuple, 0),
            context_count=context_count,
        )

    def to_serializable(self) -> dict:
        """Convert the model counts to JSON-friendly structures."""

        def join_tuple(parts: Tuple[str, ...]) -> str:
            return " ".join(parts)

        return {
            "order": self.order,
            "use_unknown_token": self.use_unknown_token,
            "vocabulary": sorted(self.vocabulary),
            "ngram_counts": {join_tuple(k): v for k, v in self.ngram_counts.items()},
            "context_counts": {
                join_tuple(k): v for k, v in self.context_counts.items()
            },
            "total_observations": self.total_observations,
        }

    @classmethod
    def from_serializable(cls, payload: dict) -> "NGramLanguageModel":
        """Rehydrate a model from ``payload`` produced by :meth:`to_serializable`."""

        model = cls(
            payload["order"],
            use_unknown_token=payload.get("use_unknown_token", True),
        )
        model.vocabulary = set(payload.get("vocabulary", []))

        ngram_payload = payload.get("ngram_counts", {})
        model.ngram_counts = Counter(
            {
                cls._split_key(key, model.order): value
                for key, value in ngram_payload.items()
            }
        )

        context_payload = payload.get("context_counts", {})
        if model.order > 1:
            model.context_counts = Counter(
                {
                    cls._split_key(key, model.order - 1): value
                    for key, value in context_payload.items()
                }
            )
        else:
            model.context_counts = Counter()

        model.total_observations = payload.get(
            "total_observations", sum(model.ngram_counts.values())
        )

        return model

    def vocabulary_size(self) -> int:
        return len(self.vocabulary)

    def _probability(self, ngram: Tuple[str, ...], smoothing: str) -> float:
        count = self.ngram_counts.get(ngram, 0)

        if self.order == 1:
            if smoothing == "laplace":
                raise ValueError("Laplace smoothing is not supported for unigrams.")
            return count / self.total_observations if self.total_observations else 0.0

        context = ngram[:-1]
        context_count = self.context_counts.get(context, 0)

        if smoothing == "laplace":
            return (count + 1) / (context_count + self.vocabulary_size())

        if count == 0 or context_count == 0:
            return 0.0

        return count / context_count

    def _validate_smoothing(self, smoothing: str) -> None:
        valid = {"none", "laplace"}
        if smoothing not in valid:
            raise ValueError(f"Unsupported smoothing strategy: {smoothing}")
        if smoothing == "laplace" and self.order == 1:
            raise ValueError("Laplace smoothing is not valid for unigram models.")

    def _generate_ngrams(self, sequence: Sequence[str]) -> Iterator[Tuple[str, ...]]:
        end = len(sequence) - self.order + 1
        for index in range(max(end, 0)):
            yield tuple(sequence[index : index + self.order])

    def _map_token(self, token: str) -> str:
        if token in self.vocabulary:
            return token
        if self.use_unknown_token and UNKNOWN_TOKEN in self.vocabulary:
            return UNKNOWN_TOKEN
        raise KeyError(f"Token {token!r} was not seen during training.")

    @staticmethod
    def _split_key(key: str, expected_length: int) -> Tuple[str, ...]:
        parts = tuple(key.split()) if key else tuple()
        if len(parts) != expected_length:
            raise ValueError(
                f"Serialized key {key!r} does not match expected length {expected_length}"
            )
        return parts
