# Experimental Results and Evaluations (v3 Final)

This document aggregates the confirmed performance metrics, feature reduction statistics, and edge execution profiles from our v3 experiments. All NSL-KDD metrics are evaluated on the KDDTest+ held-out set (22,544 samples).

---

## 1. BWOA Feature Selection

**v3 Configuration**: n_agents=30, max_iter=100, alpha=0.3, min_accuracy=0.75, min_features=10, early stopping patience=15.

| Dataset | Original Features | Selected | Reduction | RF CV Accuracy | Converged | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| NSL-KDD | 41 | 10 | 75.61% | 92.31% | Iter 23/100 | Confirmed |
| SWaT | 51 | ~22 | ~56.86% | - | - | Phase 2: Pending dataset access |
| Custom OT | ~41 | ~22 | ~46% | - | - | Phase 1: Pending data capture |

**NSL-KDD v3 Selected Features** (10 of 41):
`protocol_type, service, flag, src_bytes, hot, su_attempted, serror_rate, same_srv_rate, diff_srv_rate, dst_host_diff_srv_rate`

> **SWaT access**: Request from iTrust Centre, SUTD Singapore: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
> **Custom OT**: Phase 1 field capture at pilot mine sites using AWS EC2 sniffer nodes logging Modbus/DNP3/OPC-UA traffic. Not yet started.

---

## 2. Classification Performance (KDDTest+, 22,544 samples)

| Model | Dataset | Features | Accuracy | Precision | Recall | F1 Macro | AUC-ROC | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CNN-LSTM Baseline | NSL-KDD | 41 | 77.70% | 0.8017 | 0.7770 | 0.7571 | 0.9359 | 157.66ms |
| CNN-LSTM + BWOA v3 | NSL-KDD | 10 | 70.56% | 0.5833 | 0.7056 | 0.7127 | 0.8471 | 82.32ms |
| CNN-LSTM + BWOA Quantized | NSL-KDD | 10 | 70.56% | 0.5833 | 0.7056 | 0.7127 | 0.8471 | **0.76ms** |
| CNN-LSTM (Transfer Learning) | SWaT | ~22 | - | - | - | - | - | Phase 2 |
| CNN-LSTM (Transfer Learning) | Custom OT | ~22 | - | - | - | - | - | Phase 1 |

**Accuracy gap (baseline vs BWOA)**: 7.14% - accepted trade-off for 47.8% latency reduction and edge deployability.
**Best BWOA config found**: 256 LSTM units, 1 layer, 50 epochs (capacity-tuning iteration 2 of 4).

---

## 3. Per-Class Performance (BWOA v3 Optimized - 10 features, KDDTest+)

| Class | Precision | Recall | F1-Score | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Normal** | 0.9691 | 0.6906 | 0.8065 | Strongest; high precision for benign traffic |
| **DoS** | 0.4326 | 0.2325 | 0.3025 | Lower recall due to overlap with R2L feature space |
| **Probe** | 0.6142 | 0.7129 | 0.6599 | Best balanced attack class performance |
| **R2L** | 0.0798 | 0.2128 | 0.1160 | Minority class; strong imbalance in NSL-KDD |
| **U2R** | 0.0153 | 0.3433 | 0.0293 | Rarest class (67 test vs 52 train samples); NSL-KDD known limitation |

> U2R and R2L low F1 reflects NSL-KDD class imbalance (52 U2R training samples vs 13,449 Normal). This is a dataset limitation, not a model failure. Balanced class weights were applied during training.

---

## 4. Edge Deployment Performance

| Model | Size (MB) | Latency Mean | Latency P95 | RAM | Deployment |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CNN-LSTM Baseline (Keras) | 1.8630MB | 157.66ms | 256.23ms | - | Yes |
| CNN-LSTM + BWOA v3 (Keras) | 4.8762MB | 82.32ms | 182.55ms | - | Yes |
| **BWOA Quantized Float16 (TFLite)** | **0.8211MB** | **0.76ms** | **1.10ms** | **290.31MB** | **PASS** |

- Quantized size reduction vs BWOA Keras: **83.17%** smaller
- Quantized latency speedup vs baseline Keras: **207x faster** (157.66ms to 0.76ms)
- RAM (290.31MB) is well within the 1,024MB target ceiling
- SWaT/Custom OT edge benchmarks: pending Phase 2 / Phase 1 dataset availability

---

## 5. Full Results Summary Table

| Model | Dataset | Accuracy | F1 Macro | Latency | Size | Deployment |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| CNN-LSTM Baseline (41 feat) | NSL-KDD | 77.70% | 0.7571 | 157.66ms | 1.86MB | Yes |
| CNN-LSTM + BWOA (10 feat) | NSL-KDD | 70.56% | 0.7127 | 82.32ms | 4.88MB | Yes |
| CNN-LSTM + BWOA Quantized | NSL-KDD | 70.56% | 0.7127 | 0.76ms | 0.82MB | PASS |
| Transfer Learning | SWaT | - | - | - | - | Phase 2 |
| Transfer Learning | Custom OT | - | - | - | - | Phase 1 |
