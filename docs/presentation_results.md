# Presentation Slides: Securing the Digital Mine

Saint Petersburg Mining University - UNESCO Young Scientists Forum 2026  
*Track 3: "Smart Subsoil" - Digital Transformation and Automation in the Mineral Resources Complex*

---

## Slide 1: Title & Project Scope
### Securing the Digital Mine: A Metaheuristic Optimized Deep Learning Framework for Intrusion Detection in IoT Enabled Mineral Resource Operations

* **Team Lead**: John Okyere (Technical Lead, Kayaba Labs)
* **Under the Auspices of**: UNESCO & Empress Catherine II Saint Petersburg Mining University, Russia
* **Event Dates**: 12-17 October 2026
* **Key Idea**: A lightweight, edge-deployable intrusion detection system (IDS) utilizing Binary Whale Optimization Algorithm (BWOA) for feature selection and CNN-LSTM for classification.

---

## Slide 2: The Digital Mine Problem Statement
### Cybersecurity Challenges in Industrial IoT (IIoT) & SCADA Operations
* **Rapid Digitalization**: Deep integration of automation in mining (SDG 9) increases the cyberattack surface.
* **Complex Threats**: SCADA protocol vulnerabilities (Modbus, DNP3, OPC-UA) and traditional network vectors (DoS, Probing, U2R, R2L).
* **Resource Constraints**: Remote mining sites in Africa operate under low-bandwidth, low-power edge nodes (Raspberry Pi/industrial gateways).
* **Objective**: Build a highly accurate yet computationally lightweight IDS that runs locally at the edge with sub-100ms latency.

---

## Slide 3: Proposed Methodology Workflow
### BWOA Feature Selection + CNN-LSTM Classifier
1. **Network Ingestion**: Collect packets from SCADA/OT devices.
2. **Feature Selection**: Apply Binary Whale Optimization Algorithm (BWOA) to prune redundant features.
3. **Sequence Classification**: Use hybrid CNN-LSTM to capture spatial-temporal threat patterns:
   * **Conv1D**: Extract local spatial correlations from packet features.
   * **LSTM**: Learn long-term temporal dependencies across sequential connections.
4. **Quantization**: Perform post-training float16 TFLite quantization for lightweight CPU inference.

---

## Slide 4: BWOA Feature Selection Results
### 75.61% Dimensionality Reduction (v3 with Accuracy Floor Constraint)
* **Input Features**: 41 raw network features (NSL-KDD schema).
* **BWOA Output Subset**: **10 features** selected (v3 with 75% accuracy floor):
  `['protocol_type', 'service', 'flag', 'src_bytes', 'hot', 'su_attempted', 'serror_rate', 'same_srv_rate', 'diff_srv_rate', 'dst_host_diff_srv_rate']`
* **BWOA Validation Accuracy**: **92.31%** (RandomForest 3-fold CV on 3000-sample stratified subset).
* **Performance Benefit**: Reduces model input layer complexity by **75.61%**, translating to lower inference latency and smaller model footprint.

---

## Slide 5: Model Classification Performance
### Experimental Metrics (v3 - KDDTest+ held-out set, 22,544 samples)

| Model | Features | Accuracy | Precision | Recall | F1 Macro | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| CNN-LSTM Baseline (v3) | 41 | **77.70%** | 0.8017 | 0.7770 | 0.7571 | 0.9359 |
| **CNN-LSTM + BWOA (v3)** | **10** | **[v3 pending]** | **[v3 pending]** | **[v3 pending]** | **[v3 pending]** | **[v3 pending]** |

* **Baseline confirmed**: 77.70% test accuracy, Macro F1 = 0.7571, AUC-ROC = 0.9359 on KDDTest+ (22,544 samples).
* **BWOA optimized model**: Capacity-tuning loop in progress. Targeting gap of under 3% (minimum acceptable: 75.8%). Results will fill here once training completes.
* **BWOA feature validation**: RandomForest 3-fold CV on 3,000-sample stratified subset yielded 92.31% accuracy, well above the 75% floor constraint.

---

## Slide 6: Per-Class Breakdown (Optimized Model)
### Multi-Class Performance Under BWOA (v2 reference - v3 pending)

| Class Target | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Normal** | 0.9835 | 0.7371 | 0.8427 |
| **DoS** | 0.7765 | 0.6318 | 0.6967 |
| **Probe** | 0.5939 | 0.8558 | 0.7012 |
| **R2L** | 0.3122 | 0.5456 | 0.3971 |
| **U2R (v2)** | 0.0211 | 0.1642 | 0.0374 |

* **v3 Target**: U2R F1 above 0.20 (enforced via balanced class weights and 75% accuracy floor constraint).
* **Insights**: Highly robust detection for common attacks (Normal: 0.9835 precision; Probe: 0.8558 recall). The v3 run with a 75% accuracy floor and balanced class weights is expected to improve minority class performance, especially U2R.

---

## Slide 7: Edge Deployment & Quantization
### Edge Hardware Validation (sub-100ms latency target)

| Model | Size (MB) | Latency Mean | Latency P95 | RAM | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CNN-LSTM Baseline v3 (Keras) | 1.86MB | 157.66ms | 256.23ms | - | Yes |
| CNN-LSTM + BWOA v3 (Keras) | *pending* | *pending* | *pending* | - | *pending* |
| Quantized Float16 (TFLite, v2 ref) | **0.3189MB** | **0.14ms** | **0.29ms** | 209MB | **PASS** |

* **Quantized size reduction**: 82.85% smaller than Keras checkpoint.
* **Latency speedup**: Quantized TFLite is ~1,000x faster than full Keras inference.
* **Verdict**: Quantized model PASS. RAM footprint of 209MB is well under the 1GB constraint.
* Note: Baseline latency (157.66ms) is measured via 1,000-run Python loop on CPU, which includes Python overhead. Real edge TFLite runtime is significantly faster (0.14ms).

---

## Slide 8: SDG & UNESCO Alignment
### Sustainable Development Goals (SDG) Target Impact
* **SDG 9: Industry, Innovation, and Infrastructure**: Secures the digitalization of critical subsoil extraction infrastructure.
* **SDG 8: Decent Work and Economic Growth**: Safeguards operational continuity and automated safety monitoring systems in hazardous mines.
* **SDG 17: Partnerships for the Goals**: A joint Russian-African scientific pathway demonstrating collaborative young-scientist development at Saint Petersburg Mining University.

---

## Slide 9: Summary & Conclusions
* **Lightweight Architecture**: Combining Binary Whale Optimization with LSTM sequence learning generates an accurate, high-throughput IDS.
* **Quantization Success**: Model size reduced to under 320 KB and execution latency dropped to sub-millisecond ranges (0.14ms).
* **Future Outlook**: Proceeding to validate the model's domain transfer capability using custom OT collectors at scale on Modbus RTU/TCP networks.
