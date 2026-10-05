# Sentiment Analysis on Amazon Customer Reviews Using Parallel Deep Learning

**CSYE7105 – High Performance Parallel Machine Learning & AI**
**Team 14:** Dwij Patel and Darshak Desai
**Instructor:** Prof. Handan Liu

## Overview

This project leverages parallel computing to improve the performance of sentiment analysis on ~3.5 million Amazon customer reviews. It uses:

- **Dask** for parallelized data preprocessing (cleaning, tokenization)
- **PyTorch LSTM** for sentiment classification
- **PyTorch Distributed Data Parallel (DDP)** for multi-GPU/CPU parallel training
- **Pandas / NumPy / sklearn** for data manipulation and evaluation

## Dataset

**Source:** [Amazon Reviews – Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews/data)

- ~3.6M training reviews, ~400K test reviews (516.93 MB)
- Labels: `__label__1` (negative, 1–2 stars) / `__label__2` (positive, 4–5 stars)
- Neutral 3-star reviews excluded

**Setup:** Download `train.ft.txt` and `test.ft.txt` from Kaggle and place them in a `data/` folder:

```
sentiment_analysis_project/
├── data/
│   ├── train.ft.txt
│   └── test.ft.txt
```

## Installation

```bash
pip install torch torchvision dask[complete] pandas numpy scikit-learn matplotlib tqdm
```

## Project Structure

```
sentiment_analysis_project/
├── README.md
├── requirements.txt
├── config.py                  # Central configuration
├── preprocess.py              # Data cleaning & tokenization (Pandas baseline + Dask parallel)
├── dataset.py                 # PyTorch Dataset class
├── model.py                   # LSTM sentiment classifier
├── train_sequential.py        # Single-device training baseline
├── train_ddp.py               # Distributed Data Parallel training
├── evaluate.py                # Evaluation & metrics
├── run_experiments.py         # Full experiment pipeline with timing
└── data/
    ├── train.ft.txt
    └── test.ft.txt
```

## Usage

### 1. Preprocess the data (compares Pandas vs. Dask)

```bash
python preprocess.py
```

This generates `data/train_processed.csv` and `data/test_processed.csv`, and prints timing comparisons.

### 2. Train (sequential baseline)

```bash
python train_sequential.py
```

### 3. Train (DDP – multi-GPU)

```bash
# Single machine, 2 GPUs
torchrun --nproc_per_node=2 train_ddp.py

# Single machine, 4 GPUs
torchrun --nproc_per_node=4 train_ddp.py
```

### 4. Evaluate

```bash
python evaluate.py
```

### 5. Run full experiment pipeline

```bash
python run_experiments.py
```

## Key Results (Expected)

| Stage            | Sequential   | Parallel (Dask / DDP) | Speedup  |
|------------------|--------------|-----------------------|----------|
| Preprocessing    | ~120s        | ~40–60s (Dask)        | 2–3×     |
| Training (1 GPU) | ~3–4 hrs     | ~1–2 hrs (2 GPU DDP)  | ~2×      |

## References

- [Amazon Reviews Sentiment Analysis – LSTM (Kaggle)](https://www.kaggle.com/code/reemmuharram/amazon-reviews-sentiment-ananlysis-lstm)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Dask Documentation](https://docs.dask.org/)
