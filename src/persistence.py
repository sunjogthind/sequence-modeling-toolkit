"""Persistence helpers for model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from .models import NGramLanguageModel


def save_json(payload: dict, destination: Path) -> None:
    """Write a JSON payload to disk with UTF-8 encoding."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_json(source: Path) -> dict:
    """Read a JSON payload from disk."""

    if not source.exists():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def build_model_payload(
    models: Dict[int, NGramLanguageModel],
    *,
    metadata: dict | None = None,
) -> dict:
    """Create a serializable payload from trained models."""

    return {
        "metadata": metadata or {},
        "models": {str(order): model.to_serializable() for order, model in models.items()},
        "orders": sorted(models.keys()),
    }


def save_model_bundle(
    models: Dict[int, NGramLanguageModel],
    destination: Path,
    *,
    metadata: dict | None = None,
) -> dict:
    """Persist models alongside metadata to ``destination``."""

    payload = build_model_payload(models, metadata=metadata)
    save_json(payload, destination)
    return payload


def load_model_bundle(source: Path) -> Tuple[Dict[int, NGramLanguageModel], dict]:
    """Load models and metadata from ``source``."""

    payload = load_json(source)
    raw_models = payload.get("models", {})
    models = {
        int(order): NGramLanguageModel.from_serializable(model_payload)
        for order, model_payload in raw_models.items()
    }
    metadata = payload.get("metadata", {})
    return models, metadata
