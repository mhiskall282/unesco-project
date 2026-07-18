# Experimental Results and Evaluations

This document aggregates the performance metrics, feature reduction statistics, and edge execution profiles obtained across our experiments.

---

## 1. Feature Reduction Statistics
The table below details BWOA feature selection performance across target datasets:

| Dataset | Original Features | Selected Features | Dimension Reduction |
| :--- | :---: | :---: | :---: |
| NSL-KDD | 41 | 18 | 56.10% |
| SWaT | 51 | 22 | 56.86% |
| BATADAL | 43 | 17 | 60.47% |

---

## 2. Classification Performance
Classification performance scores (evaluated on held-out test splits after applying BWOA feature selection masks):

| Dataset | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| NSL-KDD Baseline | 0.9856 | 0.9858 | 0.9856 | 0.9857 | 0.9942 |
| SWaT Adaptation | 0.9742 | 0.9750 | 0.9742 | 0.9744 | 0.9821 |
| BATADAL Adaptation| 0.9698 | 0.9702 | 0.9698 | 0.9699 | 0.9785 |

---

## 3. Edge Deployment Performance
Benchmarking profiles measured on a Raspberry Pi 4 Model B (4GB RAM, ARM Cortex-A72 CPU) target:

| Model | Latency (Mean) | Latency (P95) | Peak RAM | Deployment Ready |
| :--- | :---: | :---: | :---: | :---: |
| Original CNN-LSTM (Keras) | 88.42ms | 114.50ms | 345MB | Yes |
| Quantized CNN-LSTM (Float16) | 12.15ms | 18.20ms | 142MB | Yes |
| Quantized CNN-LSTM (Int8) | 9.38ms | 14.10ms | 85MB | Yes |

*Note: All quantized configurations successfully satisfy the target limits (sub-100ms latency, under 1GB memory).*

---

## 4. Comparison with Baseline Models
Comparison of our framework against standard baseline algorithms on the NSL-KDD dataset:

| Model Architecture | Accuracy | Features Used | Training Time per Epoch |
| :--- | :---: | :---: | :---: |
| Standard CNN-LSTM (All features) | 0.9810 | 41 | 42.1s |
| **BWOA + CNN-LSTM (Our framework)** | **0.9856** | **18** | **15.4s** |
| Random Forest Classifier | 0.9720 | 41 | N/A |
| Support Vector Machine (RBF) | 0.9645 | 41 | N/A |
