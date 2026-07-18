# Experimental Results and Evaluations (v3)

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
| CNN-LSTM (all features) | NSL-KDD | 0.7770 | 0.8017 | 0.7770 | 0.7571 | 0.9359 |
| CNN-LSTM + BWOA (v3) | NSL-KDD | 0.7056 | 0.7797 | 0.7056 | 0.7127 | 0.8758 |

---

## 3. Edge Deployment Performance
Benchmarking profiles measured on a Raspberry Pi 4 Model B (4GB RAM, ARM Cortex-A72 CPU) target:

| Model | Latency Mean (ms) | Latency P95 (ms) | Model Size (MB) | Deployment Ready |
| :--- | :---: | :---: | :---: | :---: |
| Original CNN-LSTM (Keras) | 157.66ms | 256.23ms | 1.86MB | Yes |
| Quantized CNN-LSTM (Float16) | 0.76ms | 1.10ms | 0.8211MB | Yes |

---

## 4. Per-Class Performance (BWOA Optimized Model)
Weighted and class-specific metrics breakdown for the optimized BWOA classifier on the test dataset:

| Class | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: |
| Normal | 0.9689 | 0.6839 | 0.8018 |
| DoS | 0.7514 | 0.8904 | 0.8150 |
| Probe | 0.5488 | 0.7080 | 0.6183 |
| R2L | 0.5971 | 0.1449 | 0.2332 |
| U2R | 0.0134 | 0.3881 | 0.0258 |
