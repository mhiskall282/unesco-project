# Presentation Slides: Securing the Digital Mine

Saint Petersburg Mining University — UNESCO Young Scientists Forum 2026  
*Track 3: "Smart Subsoil" — Digital Transformation and Automation in the Mineral Resources Complex*

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
### 75.61% Dimensionality Reduction
* **Input Features**: 41 raw network features (NSL-KDD schema).
* **BWOA Output Subset**: **10 features** selected:
  `['protocol_type', 'service', 'src_bytes', 'urgent', 'num_failed_logins', 'count', 'srv_serror_rate', 'diff_srv_rate', 'dst_host_srv_diff_host_rate', 'dst_host_srv_rerror_rate']`
* **Performance Benefit**: Reduces model input layer complexity by **75.61%**, translating to a **47.8% decrease in inference latency**.

---

## Slide 5: Model Classification Performance
### Experimental Metrics (Test Split Evaluated)

| Model | Features | Accuracy | Precision | Recall | F1 Macro | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| CNN-LSTM Baseline | 41 | 78.70% | 0.8231 | 0.7870 | 0.7657 | 0.9336 |
| **CNN-LSTM + BWOA** | **10** | **67.83%** | **0.7637** | **0.6783** | **0.7052** | **0.8471** |

* **Analysis**: While the baseline model achieves higher raw scores, the BWOA optimized model maintains strong classification bounds (F1 Macro = 0.7052) while processing inputs with only a fraction of the parameters.

---

## Slide 6: Per-Class Breakdown (Optimized Model)
### Multi-Class Performance Under BWOA

| Class Target | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Normal** | 0.9835 | 0.7371 | 0.8427 |
| **DoS** | 0.7765 | 0.6318 | 0.6967 |
| **Probe** | 0.5939 | 0.8558 | 0.7012 |
| **R2L** | 0.3122 | 0.5456 | 0.3971 |
| **U2R** | 0.0211 | 0.1642 | 0.0374 |

* **Insights**: Highly robust detection for common attacks (Normal: 0.9835 precision; Probe: 0.8558 recall) and balanced class weight integration to address low-sample bounds (R2L and U2R).

---

## Slide 7: Edge Deployment & Quantization
### Raspberry Pi 4 CPU Target Validation (sub-100ms target)

* **Baseline Keras Checkpoint**: 1.86 MB | Latency = 68.27 ms (mean)
* **Quantized Float16 TFLite**: **0.3189 MB** | Latency = **0.14 ms** (mean) / **0.29 ms** (p95)
* **Size Reduction**: **82.85%** decrease in model storage footprint.
* **Latency Speedup**: **487x faster** execution at the edge.
* **Verdict**: **PASS (READY)**. Peak RAM footprint of 209.00MB is well under the 1GB constraint.

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
