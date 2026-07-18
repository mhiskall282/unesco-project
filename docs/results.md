# Experimental Results and Evaluations (v3)

This document aggregates the performance metrics, feature reduction statistics, and edge execution profiles obtained across our experiments. All numbers are evaluated on the KDDTest+ held-out set (22,544 samples) unless noted otherwise.

---

## 1. BWOA Feature Selection

**v3 Configuration**: n_agents=30, max_iter=100, alpha=0.3 (70% accuracy weight / 30% feature reduction), min_accuracy=0.75 (floor constraint), min_features=10, early stopping patience=15.

| Dataset | Original Features | Selected Features | Reduction | RF CV Accuracy | Converged At |
| :--- | :---: | :---: | :---: | :---: | :---: |
| NSL-KDD | 41 | 10 | 75.61% | 92.31% | Iter 23/100 |
| SWaT | 51 | 22 | 56.86% | N/A | TBD |
| BATADAL | 43 | 17 | 60.47% | N/A | TBD |

**NSL-KDD v3 Selected Features** (10 of 41):
`protocol_type, service, flag, src_bytes, hot, su_attempted, serror_rate, same_srv_rate, diff_srv_rate, dst_host_diff_srv_rate`

---

## 2. Classification Performance (KDDTest+, 22,544 samples)

| Model | Features | Accuracy | Precision | Recall | F1 Macro | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| CNN-LSTM Baseline (v3) | 41 | 0.7770 | 0.8017 | 0.7770 | 0.7571 | 0.9359 |
| CNN-LSTM + BWOA (v3) | 10 | *pending* | *pending* | *pending* | *pending* | *pending* |

> Baseline v3 confirmed from `logs/baseline_v3_metrics.json`. Optimized model training in progress (capacity-tuning loop, epoch 27/50 of config 1).

**Accuracy gap target**: within 3% of baseline (77.70%). Minimum acceptable: 75.8%.

---

## 3. Edge Deployment Performance

Benchmarking measured on CPU (simulated Raspberry Pi 4 profile, 1000-run latency loop):

| Model | Latency Mean (ms) | Latency P95 (ms) | Model Size (MB) | Deployment Ready |
| :--- | :---: | :---: | :---: | :---: |
| CNN-LSTM Baseline v3 (Keras) | 157.66ms | 256.23ms | 1.86MB | Yes |
| CNN-LSTM + BWOA v3 (Keras) | *pending* | *pending* | *pending* | *pending* |
| Quantized Float16 (TFLite, v2 ref) | 0.14ms | 0.29ms | 0.3189MB | Yes |

> Latency for v2 quantized model is included as reference. v3 quantized benchmark will update once optimized model saves.

---

## 4. Per-Class Performance

### Baseline v3 - CNN-LSTM (41 features) on KDDTest+

| Class | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: |
| Normal | *from confusion matrix* | *pending* | *pending* |
| DoS | *pending* | *pending* | *pending* |
| Probe | *pending* | *pending* | *pending* |
| R2L | *pending* | *pending* | *pending* |
| U2R | *pending* | *pending* | *pending* |

### BWOA v3 Optimized (10 features) on KDDTest+

| Class | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: |
| Normal | *pending* | *pending* | *pending* |
| DoS | *pending* | *pending* | *pending* |
| Probe | *pending* | *pending* | *pending* |
| R2L | *pending* | *pending* | *pending* |
| U2R | *pending* | *pending* | *pending* |

> Note: U2R is the rarest attack class in NSL-KDD (67 test samples vs 13,449 Normal). Lower F1 for U2R reflects dataset class imbalance, not model failure. Balanced class weights applied during training.

---

## 5. Comparison with Baseline Architectures

| Model Architecture | Accuracy | Features Used | Notes |
| :--- | :---: | :---: | :--- |
| CNN-LSTM Baseline (v3) | 0.7770 | 41 | Full feature set, confirmed on KDDTest+ |
| **CNN-LSTM + BWOA v3 (ours)** | **pending** | **10** | BWOA-optimized, capacity-tuning in progress |
| Random Forest Classifier | 0.9231 | 10 | Used as BWOA fitness proxy (RF CV val subset) |
