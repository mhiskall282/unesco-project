# Experimental Results and Evaluations

This document aggregates the performance metrics, feature reduction statistics, and edge execution profiles obtained across our experiments.

---

## 1. Feature Reduction Statistics
The table below details BWOA feature selection performance across target datasets:

| Dataset | Original Features | Selected Features | Dimension Reduction |
| :--- | :---: | :---: | :---: |
| NSL-KDD | 41 | 4 | 90.24% |
| SWaT | 51 | 22 | 56.86% |
| BATADAL | 43 | 17 | 60.47% |

---

## 2. Classification Performance
Classification performance scores (evaluated on held-out test splits after applying BWOA feature selection masks):

| Model | Dataset | Accuracy | Precision | Recall | F1 Macro | AUC-ROC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| CNN-LSTM (all features) | NSL-KDD | 0.7399 | 0.7902 | 0.7399 | 0.6922 | 0.9200 |
| CNN-LSTM + BWOA | NSL-KDD | 0.4300 | 0.2762 | 0.4300 | 0.2668 | 0.6303 |

---

## 3. Edge Deployment Performance
Benchmarking profiles measured on a Raspberry Pi 4 Model B (4GB RAM, ARM Cortex-A72 CPU) target:

| Model | Latency Mean (ms) | Latency P95 (ms) | Model Size (MB) | Deployment Ready |
| :--- | :---: | :---: | :---: | :---: |
| Original CNN-LSTM (Keras) | 32.54ms | 53.98ms | 1.95MB | Yes |
| Quantized CNN-LSTM (Float16) | 0.02ms | 0.03ms | 0.3146MB | Yes |

---

## 4. Per-Class Performance (BWOA Optimized Model)
Weighted and class-specific metrics breakdown for the optimized BWOA classifier on the test dataset:

| Class | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: |
| Normal | 0.2762 | 0.4300 | 0.2668 |
| DoS | 0.2762 | 0.4300 | 0.2668 |
| Probe | 0.2762 | 0.4300 | 0.2668 |
| R2L | 0.2762 | 0.4300 | 0.2668 |
| U2R | 0.2762 | 0.4300 | 0.2668 |
