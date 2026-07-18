# Experimental Results and Evaluations (v2)

This document aggregates the performance metrics, feature reduction statistics, and edge execution profiles obtained across our experiments.

---

## 1. Feature Reduction Statistics
The table below details BWOA feature selection performance across target datasets:

| Dataset | Original Features | Selected Features | Dimension Reduction |
| :--- | :---: | :---: | :---: |
| NSL-KDD | 41 | 10 | 75.61% |
| SWaT | 51 | 22 | 56.86% |
| BATADAL | 43 | 17 | 60.47% |

---

## 2. Classification Performance
Classification performance scores (evaluated on held-out test splits after applying BWOA feature selection masks):

| Model | Dataset | Accuracy | Precision | Recall | F1 Macro | AUC-ROC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| CNN-LSTM (all features) | NSL-KDD | 0.7870 | 0.8231 | 0.7870 | 0.7657 | 0.9336 |
| CNN-LSTM + BWOA | NSL-KDD | 0.6783 | 0.7637 | 0.6783 | 0.7052 | 0.8471 |

---

## 3. Edge Deployment Performance
Benchmarking profiles measured on a Raspberry Pi 4 Model B (4GB RAM, ARM Cortex-A72 CPU) target:

| Model | Latency Mean (ms) | Latency P95 (ms) | Model Size (MB) | Deployment Ready |
| :--- | :---: | :---: | :---: | :---: |
| Original CNN-LSTM (Keras) | 68.27ms | 89.64ms | 1.86MB | Yes |
| Quantized CNN-LSTM (Float16) | 0.14ms | 0.29ms | 0.3189MB | Yes |

---

## 4. Per-Class Performance (BWOA Optimized Model)
Weighted and class-specific metrics breakdown for the optimized BWOA classifier on the test dataset:

| Class | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: |
| Normal | 0.9835 | 0.7371 | 0.8427 |
| DoS | 0.7765 | 0.6318 | 0.6967 |
| Probe | 0.5939 | 0.8558 | 0.7012 |
| R2L | 0.3122 | 0.5456 | 0.3971 |
| U2R | 0.0211 | 0.1642 | 0.0374 |
