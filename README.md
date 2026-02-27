# BERT 20 Newsgroups Classification with Confidence Calibration

## Overview
This project builds a robust BERT-based text classification system using the 20 Newsgroups dataset.

Beyond standard classification, it explores:

- Confidence estimation
- Temperature scaling calibration
- Domain shift evaluation
- Selective prediction

## Model Architecture
- Pretrained: bert-base-uncased
- Max length: 256
- Head-Tail truncation
- Dropout + Linear classification head
- CrossEntropyLoss with label smoothing

## Metrics
- Accuracy
- Precision / Recall / F1
- Expected Calibration Error (ECE)
- Negative Log-Likelihood (NLL)

## Advanced Features
- Mixed Precision Training (AMP)
- Temperature Scaling (Validation-based)
- Confidence Thresholding for "Other" class
- Domain Shift Evaluation

## Tech Stack
- Python
- PyTorch
- HuggingFace Transformers
- Scikit-learn
- Matplotlib
