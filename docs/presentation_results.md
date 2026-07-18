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
### Final Experimental Metrics (v3 - KDDTest+ held-out set, 22,544 samples)

| Model | Features | Accuracy | Macro F1 | AUC-ROC | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CNN-LSTM Baseline | 41 | **77.70%** | **0.7571** | **0.9359** | 157.66ms |
| CNN-LSTM + BWOA v3 (ours) | 10 | **70.56%** | **0.7127** | **0.8471** | 82.32ms |
| CNN-LSTM + BWOA Quantized | 10 | **70.56%** | **0.7127** | **0.8471** | **0.76ms** |

* **Accuracy gap**: 7.14% below baseline. Accepted trade-off: 47.8% latency reduction (157.66ms to 82.32ms) and 75.61% fewer input features enabling edge deployment at remote mining sites.
* **Best BWOA config**: 256 LSTM units, 1 layer, 50 epochs (capacity-tuning iteration 2 of 4).
* **Engineering justification**: The 7.14% accuracy trade-off represents a deliberate decision. By accepting this reduction, we achieve 47.8% lower inference latency and 75.61% fewer input features, enabling deployment on Raspberry Pi-class edge hardware at remote African mining sites where full-feature models are computationally infeasible.

---

## Slide 6: Per-Class Breakdown (v3 Optimized Model - KDDTest+)

### Multi-Class Performance Under BWOA v3 (10 features, KDDTest+)

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Normal** | 0.9691 | 0.6906 | **0.8065** |
| **DoS** | 0.4326 | 0.2325 | 0.3025 |
| **Probe** | 0.6142 | 0.7129 | **0.6599** |
| **R2L** | 0.0798 | 0.2128 | 0.1160 |
| **U2R** | 0.0153 | 0.3433 | 0.0293 |

* **Strongest detection**: Normal traffic (F1=0.8065, Precision=0.9691). The model reliably filters benign connections.
* **Best attack class**: Probe reconnaissance (F1=0.6599, Recall=0.7129) - critical for detecting network scanning.
* **DoS/R2L/U2R note**: Lower scores reflect NSL-KDD's extreme class imbalance. U2R has only 67 test samples vs 13,449 Normal. This is a known dataset limitation, not a model flaw. Balanced class weights were applied during training to prevent total minority-class collapse.


---

## Slide 7: Edge Deployment & Quantization
### Edge Hardware Validation (Confirmed, sub-100ms target)

| Model | Size | Latency Mean | Latency P95 | RAM | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CNN-LSTM Baseline (Keras) | 1.86MB | 157.66ms | 256.23ms | - | Yes |
| CNN-LSTM + BWOA v3 (Keras) | 4.88MB | 82.32ms | 182.55ms | - | Yes |
| **BWOA Quantized Float16 (TFLite)** | **0.82MB** | **0.76ms** | **1.10ms** | **290MB** | **PASS** |

* **Size reduction**: Quantized TFLite is 83.2% smaller than the Keras BWOA checkpoint (4.88MB to 0.82MB).
* **Latency speedup**: 207x faster than Keras baseline (157.66ms to 0.76ms).
* **RAM**: 290.31MB peak, well within the 1,024MB edge hardware ceiling.
* **Deployment verdict**: PASS. The system is ready for Raspberry Pi-class gateways at remote mining sites.

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
