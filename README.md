# N-Gram Phonetic Language Model

This project implements a phonetic sequence modeling pipeline that transforms raw CHILDES speech transcripts into training corpora and trains n-gram language models from scratch. A unified CLI covers corpus preparation, multi-order model training, and statistical evaluation with reproducible experiments.

## Why this project matters

Modern conversational AI systems rely on robust acoustic priors. This project demonstrates how to build those priors with careful data engineering and statistical modeling:

- **End-to-end pipeline** – deterministic data ingestion from nested corpora, stratified splits with seeding, and artifact tracking for auditability.
- **Production-hardened n-gram stack** – extensible `NGramLanguageModel` abstraction with Laplace smoothing, flexible context windows, and cross-order persistence.
- **Enterprise-grade evaluation** – perplexity analytics with automatic OOV handling and early warnings when model support is insufficient, ensuring reliable deployment metrics.

## Core capabilities

1. **Corpus normalization** – consolidate raw CHILDES phoneme sequences into curated train/dev assets with automatic quality filters.
2. **Model factory** – train unigram, bigram, trigram (or higher) models in one pass and persist rich metadata for experiment traceability.
3. **Evaluation console** – compute perplexity under multiple smoothing schemes to baseline language-model fitness pre-deployment.

## Quickstart

```bash
# 1. Prepare curated splits from a transformed CHILDES directory
python3 src/main.py split \
    --input data/transformed \
    --train-out artifacts/training.txt \
    --dev-out artifacts/dev.txt \
    --train-ratio 0.8 \
    --seed 42

# 2. Train multi-order language models
python3 src/main.py train \
    --input artifacts/training.txt \
    --output artifacts/models.json \
    --orders 1 2 3

# 3. Evaluate perplexity across smoothing strategies
python3 src/main.py eval \
    --model-file artifacts/models.json \
    --order 3 \
    --data artifacts/dev.txt \
    --smoothing laplace
```

Each command surfaces structured logging to make experiment reproduction trivial when presenting results to stakeholders.

## Repository layout

```
project_3/
├─ README.md                     # Strategic overview & quickstart
├─ docs/
│  └─ portfolio_blurb.md         # Resume-ready project pitch
├─ src/
│  ├─ __init__.py                # Marks package for Python imports
│  ├─ main.py                    # Unified CLI entrypoint
│  ├─ data_pipeline.py           # Corpus ingestion & deterministic splits
│  ├─ models.py                  # NGramLanguageModel implementation
│  └─ persistence.py             # JSON artifact serializers/deserializers
└─ data/
   └─ sample_dev.txt             # Lightweight sample for smoke testing
```

## Extending the platform

- **KenLM interop** – export `.arpa` assets directly from n-gram counts to plug into production ASR pipelines.
- **Neural priors** – blend n-gram probabilities with transformer-based acoustic encoders for hybrid inference.
- **Observability** – push perplexity trends into Prometheus/Grafana for live monitoring of deployed conversational agents.

## License & attribution

The toolkit is an original refactoring inspired by academic coursework. All code is implemented from scratch without third-party NLP libraries, making it safe to showcase publicly and to extend within enterprise environments.
