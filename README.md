# PocketNarrator: Efficient Story Generation with Small Language Models

**Status:** Active Development 🚀

PocketNarrator is a research project for the "Efficient Methods in Machine Learning" course (Master Project, WS25/26) at the University of Hamburg. It focuses on building and evaluating small language models for narrative generation using the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Models](#supported-models)
- [Directory Structure](#directory-structure)
- [Requirements](#requirements)
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Set Up Environment](#2-set-up-environment)
  - [3. Install Python Dependencies](#3-install-python-dependencies)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Generation](#generation)
- [Project Structure](#project-structure)
- [Team](#team)

## Overview

PocketNarrator is a systematic investigation into the architecture and components of small language models for efficient narrative generation. Our goal is to understand trade-offs between different architectural choices in terms of performance, computational efficiency, and output quality.

The project implements multiple model architectures from scratch using PyTorch, trained on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories)—a clean, restricted-domain collection of children's stories ideal for training efficient models. Our models are designed for next-token prediction and story continuation tasks.

## Features

- **Multiple model architectures** supporting N-gram and Transformer models
- **Comprehensive evaluation metrics** including BLEU, ROUGE, perplexity, distinct-n, text quality, and noun carryover analysis
- **Flexible tokenization** with BPE and character-level tokenizers
- **W&B integration** for experiment tracking and visualization
- **Production-ready evaluation pipeline** with model comparison and dataset analysis tools
- **Clean, modular codebase** with abstract base classes for extensibility

## Supported Models

- **N-gram Model**: Lightweight baseline model for quick experiments
- **Transformer Model**: Custom decoder-only transformer architecture with configurable attention mechanisms

## Directory Structure

```
pocket-narrator/
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies
├── pyproject.toml                       # Project configuration
├── pytest.ini                           # Pytest configuration
│
├── configs/                             # Configuration files for models and training
│   ├── base_config.yaml
│   ├── evaluation/                      # Evaluation configs
│   ├── models/                          # Model-specific configs
│   ├── tokenizers/                      # Tokenizer configs
│   └── training/                        # Training configs
│
├── data/                                # Datasets (raw and processed)
│   ├── raw/                             # Original dataset files
│   └── processed/                       # Processed datasets
│
├── models/                              # Trained model checkpoints
│   ├── ngram/
│   ├── transformer/
│   └── cool_models/
│
├── notebooks/                           # Jupyter notebooks for exploration
│
├── pocket_narrator/                     # Main package source code
│   ├── __init__.py
│   ├── models/                          # Model architectures
│   │   ├── base_model.py                # Abstract base class
│   │   ├── ngram_model.py               # N-gram implementation
│   │   ├── components/                  # Reusable components
│   │   │   ├── positional_encoding.py
│   │   │   └── base_pos_encoding.py
│   │   └── transformers/                # Transformer architecture
│   │       ├── model.py
│   │       ├── transformer_block.py
│   │       ├── attention.py
│   │       └── base_attention.py
│   │
│   ├── tokenizers/                      # Tokenization implementations
│   │   ├── base_tokenizer.py
│   │   ├── bpe_tokenizer.py
│   │   └── character_tokenizer.py
│   │
│   ├── trainers/                        # Model trainers
│   │   ├── base_trainer.py
│   │   ├── ngram_trainer.py
│   │   └── transformer_trainer.py
│   │
│   ├── data_loader.py                   # Data loading utilities
│   ├── evaluate.py                      # Evaluation metrics
│   ├── text_quality.py                  # Text quality evaluation
│   ├── noun_carryover.py                # Noun carryover metrics
│   └── gemini_api.py                    # LLM-based evaluation
│
├── scripts/                             # Standalone execution scripts
│   ├── train.py                         # Main training script
│   ├── generate.py                      # Text generation
│   ├── evaluate_model.py                # Single model evaluation
│   ├── evaluate_dataset_comprehensive.py # Dataset evaluation
│   ├── preprocess.py                    # Data preprocessing
│   ├── fetch_tinystories.py             # Dataset download
│   └── …
│
├── tests/                               # Unit and integration tests
│
├── tokenizers/                          # Saved tokenizer artifacts
│
├── results/                             # Evaluation results (JSON)
│
└── wandb/                               # W&B experiment tracking
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas
- Hugging Face Transformers & Datasets
- wandb (for experiment tracking)
- Optional: spacy, sentence-transformers, google-genai (for advanced evaluation)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/bschink/pocket-narrator.git
cd pocket-narrator
```

### 2. Set Up Environment

```bash
# Create conda environment with PyTorch
conda create -n pocket-narrator python=3.10 pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# Activate the conda environment
conda activate pocket-narrator
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a model using the training script:

```bash
# Train with default config
python scripts/train.py

# Train with specific config
python scripts/train.py --config_path configs/training/train_tinystories_1M.yaml

# Key arguments:
#   --config_path: Path to training config YAML
#   --model_type: ngram or transformer
#   --dataset_path: Path to dataset
#   --output_dir: Directory to save models
#   --epochs: Number of training epochs
#   --batch_size: Batch size for training
```

### Evaluation

Evaluate a trained model on a dataset:

```bash
# Evaluate single model with comprehensive metrics
python scripts/evaluate_model.py \
    --model_path models/transformer/transformer_model.pth \
    --model_type transformer \
    --dataset_path data/test_dataset.txt

# Evaluate dataset without a model (text quality, distinct-n, etc.)
python scripts/evaluate_dataset_comprehensive.py \
    --dataset_path data/validation.txt \
    --dataset_name "TinyStories Validation"
```

### Generation

Generate text continuations with a trained model:

```bash
# Generate with default prompt
python scripts/generate.py \
    --model_path models/transformer/transformer_model.pth \
    --model_type transformer

# Generate with custom prompt
python scripts/generate.py \
    --model_path models/transformer/transformer_model.pth \
    --model_type transformer \
    --prompt "A girl went to the" \
    --max_length 100 \
    --temperature 0.7
```

## Project Structure

### Core Components

- **Models**: Extensible model implementations (N-gram, Transformer)
- **Tokenizers**: Multiple tokenization strategies (BPE, character-level)
- **Trainers**: Trainer classes for different architectures
- **Evaluation**: Comprehensive metrics (BLEU, ROUGE, perplexity, text quality, LLM judgments)

### Key Features

- **Configuration-driven**: All experiments defined in YAML configs
- **Experiment tracking**: W&B integration for logging and visualization
- **Modular design**: Easy to add new models, tokenizers, or evaluation metrics
- **Well-tested**: Unit and integration tests for core functionality

## Team

Asiya Yumna, Kosar Hazrati & Benedikt Schink
