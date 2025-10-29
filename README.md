# PocketNarrator: Generating Children Stories Efficiently

**Status:** In Progress 🚧

PocketNarrator is a project for the "Efficient Methods in Machine Learning" course (Master Project, WS25/26) at the University of Hamburg.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Requirements](#requirements)
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Set Up Environment](#2-set-up-environment)
  - [3. Install Python Dependencies](#3-install-python-dependencies)
- [Usage](#usage)
  - [Training](#training)
  - [Generation](#generation)
- [Team](#team)

## Overview

This project is a systematic investigation into the architecture and components of small language models for efficient narrative generation. Our primary goal is to develop a deep understanding of the trade-offs between different architectural choices in terms of performance, computational efficiency and output quality, rather than simply building a model that can generate stories.

To achieve this, we will implement language models from scratch using PyTorch. The project will be centered around the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories), a clean, restricted-domain dataset ideal for training capable small models without requiring massive computational resources. One thing to note: This dataset contains a collection of short children stories which we will make use of to build a simple language model with the goal of completing sentences (next token prediction). Firstly, we aim to replicate and optimise the paper corresponding to the dataset [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) (Eldan, et al.) by using the Transformer architecture proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani, et al.). If time permits, we will compare the Transformer architecture with Mamba, a state-space model proposed in [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (Gu, et al.).

## Features

- **No features yet**: the features will be added alongside the course of the project

## Directory Structure

```
pocket-narrator/
├── README.md
├── requirements.txt
├── configs/                 # Configuration files for experiments
│   ├── base_config.yaml
├── data/                    # Raw and processed datasets
│   ├── .gitkeep
├── models/                  # Saved model checkpoints
│   ├── .gitkeep
├── notebooks/               # Exploratory data analysis and experimentation
│   └── data_exploration.ipynb
├── pocket_narrator/         # Source code for the PocketNarrator package
│   ├── __init__.py
│   ├── tokenizer.py         # Handles text tokenization
│   ├── data_loader.py       # Loads and preprocesses data for the model
│   ├── model.py             # The core model architecture
│   ├── evaluate.py          # Functions for evaluating model performance
│   └── utils.py             # Helper functions used across the package
├── scripts/                 # Standalone scripts for execution
│   ├── train.py             # Script to train the model
│   └── generate.py          # Script to generate text with a trained model
├── tests/                   # Unit and integration tests
│   └── test_*.py            # Individual test files
```

## Requirements

- none yet

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/bschink/pocket-narrator.git
cd pocket-narrator
```

### 2. Set Up Environment

```bash
# Create conda environment
conda create -n pocket-narrator

# Activate the conda environment
conda activate pocket-narrator
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
# not yet functional
python scripts/train.py
```

### Generation

```bash
# not yet functional
python scripts/generate.py
```

## Team

Asiya Yumna, Kosar Hazrati & Benedikt Schink
