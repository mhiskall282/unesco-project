# Project Context: Securing the Digital Mine

## Overview
This repository contains the codebase and research artifacts for "Securing the Digital Mine: A Metaheuristic Optimized Deep Learning Framework for Intrusion Detection in IoT Enabled Mineral Resource Operations".

### Competition & Track
- **Competition**: Russian-African Forum-Contest of Young Scientists 2026, "Future Engineers of the World: The Foundation of Sustainable Development", held under the auspices of UNESCO.
- **Host**: Empress Catherine II Saint Petersburg Mining University, Russia.
- **Event Dates**: 12 to 17 October 2026.
- **Track**: Track 3 "Smart Subsoil", focusing on Digital Transformation and Automation in the Mineral Resources Complex.

### Team Lead
- **Name**: John Okyere
- **Role**: BSc ICT Education candidate (University of Education, Winneba, Ghana).
- **Affiliations**: Co-founder and Technical Lead at Kayaba Labs; ICP Ambassador. John has trained over 600 developers across Ghana and internationally.

---

## Technical Stack and Architecture
The framework leverages a hybrid metaheuristic deep learning architecture designed to detect cyberattacks and intrusions in modern industrial control systems (ICS) and Internet of Things (IoT) environments in mining operations.

### Key Components
1. **Feature Selection (BWOA)**:
   A Binary Whale Optimization Algorithm (BWOA) is utilized to perform intelligent feature selection on network traffic headers and payloads, reducing dimensionality while maximizing classification accuracy.
2. **Deep Learning Classifier (CNN-LSTM)**:
   A hybrid model consisting of a 1D Convolutional Neural Network (CNN) for spatial feature extraction followed by a Long Short-Term Memory (LSTM) network for capturing temporal sequence correlations.
3. **Data Collection Pipeline**:
   Deployed on AWS EC2 instances, capturing network traffic and generating flow-based features using CICFlowMeter.
4. **Edge Deployment Target**:
   Raspberry Pi-class hardware with a focus on resource efficiency (maximum 1GB RAM) and low-latency performance (sub-100ms detection latency).

### Software Stack
- **Programming Language**: Python 3.11+
- **Deep Learning Framework**: TensorFlow/Keras
- **Machine Learning & Data Processing**: scikit-learn, NumPy, pandas
- **Network Feature Extraction**: CICFlowMeter

---

## Three-Phase Roadmap
The research and development lifecycle is divided into three key phases:

1. **Phase 1: Data Collection & Labeling (Months 1 to 6)**
   - Deploys CICFlowMeter traffic capture at pilot mining sites.
   - Captures industrial control protocols and network traffic, creating a labeled custom OT dataset.
2. **Phase 2: Model Adaptation & Validation (Months 4 to 10)**
   - Implements BWOA for feature selection.
   - Adapts and retrains the CNN-LSTM deep learning classifier.
   - Validates baseline performance against NSL-KDD, SWaT, and BATADAL datasets.
3. **Phase 3: Edge Deployment & Evaluation (Months 8 to 14)**
   - Optimizes and quantizes the trained model for Raspberry Pi-class devices.
   - Benchmarks detection latency, CPU/memory consumption, and localized execution feasibility.

---

## Dataset Descriptions
The framework is evaluated across benchmark and site-specific datasets:

- **NSL-KDD**:
  An improved version of the classic KDD'99 network intrusion dataset. It is used as the initial baseline validation target for network traffic classification.
- **SWaT (Secure Water Treatment)**:
  A real-world testbed dataset representing water treatment operations. It provides realistic industrial control system (ICS) traffic and physical sensor readings under cyberattacks.
- **BATADAL (Battle of the Attack Detection Algorithms)**:
  A benchmark dataset focusing on detecting cyberattacks on water distribution systems, capturing SCADA telemetry and anomalous operational flows.
- **Custom OT Dataset**:
  Operational technology network traffic captured directly from pilot mining sites (SCADA, IoT sensors, cloud-connected digital twins) using the AWS-CICFlowMeter pipeline.

---

## Sustainable Development Goals (SDGs) Alignment
This research project aligns with the United Nations SDGs to drive sustainable industrial growth:

- **SDG 9: Industry, Innovation, and Infrastructure**
  By securing critical infrastructure (mining SCADA and IIoT networks) against cyber threats, this project fosters resilient digital transformation and innovation in subsoil operations.
- **SDG 8: Decent Work and Economic Growth**
  Intrusion detection systems prevent malicious operational disruptions, thereby safeguarding workers, physical assets, and productivity in mining environments.
- **SDG 17: Partnerships for the Goals**
  Cooperative scientific development and knowledge exchange between Ghana (University of Ghana, Kayaba Labs) and Russia (Saint Petersburg Mining University) under UNESCO's auspices.

---

## Academic References
- Mirjalili, S., & Lewis, A. (2016). The Whale Optimization Algorithm. *Advances in Engineering Software*, 95, 51:67. https://doi.org/10.1016/j.advengsoft.2016.01.008
- Kheddar, H., Himeur, Y., & Awad, A. (2023). Deep transfer learning for industrial control systems intrusion detection: A systematic review. *Journal of Network and Computer Applications*, 214, 103597. https://doi.org/10.1016/j.jnca.2023.103597
- Alanazi, M., Mahmood, A., & Chowdhury, M. J. M. (2022). SCADA vulnerabilities and attacks: A systematic review. *Computers & Security*, 125, 103028. https://doi.org/10.1016/j.cose.2022.103028
- Almomani, A., Akour, M., & Habeb, M. (2025). Cyberattack detection for SCADA systems in IIoT environments. *Symmetry*, 17(4), 480. https://doi.org/10.3390/sym17040480
- Krishnaveni, S., et al. (2025). Hybrid metaheuristic intrusion detection system for wireless sensor networks in industrial environments. *Cluster Computing*, 28, 5248. https://doi.org/10.1007/s10586-024-04612-x
- Anand, P., & Arul, R. (2024). Whale Optimization Algorithm enhanced LSTM for intrusion detection. *Cryptography*, 8(4), 73. https://doi.org/10.3390/cryptography8040073

---

## Coding Conventions
All source code and documentation must strictly adhere to the following conventions:
- **Language**: Python 3.11 or higher.
- **Style Guide**: PEP8 compliance is mandatory.
- **Type Annotations**: Type hints must be used for all function signatures and variable declarations where applicable.
- **Documentation**: All modules, classes, and functions must have Google-style docstrings.
- **Punctuation Constraint**: Em dashes (—) are strictly prohibited across all documentation, code comments, and project markdown files.
