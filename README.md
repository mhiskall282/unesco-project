# Securing the Digital Mine

> A Metaheuristic Optimized Deep Learning Framework for Intrusion Detection in IoT Enabled Mineral Resource Operations

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![UNESCO Forum](https://img.shields.io/badge/UNESCO-Russian--African%20Forum%202026-blue.svg)](https://youthafrica.spmi.ru)
[![Track](https://img.shields.io/badge/Track%203-Smart%20Subsoil-green.svg)](https://youthafrica.spmi.ru/en/participants)

African and Russian mining operations are digitalizing faster than their cybersecurity posture can keep pace. This project adapts a Binary Whale Optimization Algorithm combined with a CNN-LSTM deep learning classifier, validated on NSL-KDD, toward the distinct traffic patterns of mining IoT and SCADA infrastructure. The framework is purpose-built for edge deployment in resource-constrained African mining environments.

**Competition:** Russian-African Forum-Contest of Young Scientists 2026, Saint Petersburg Mining University, Russia  
**Track:** Track 3 "Smart Subsoil", focusing on Digital Transformation and Automation in the Mineral Resources Complex  
**Event Dates:** 12 to 17 October 2026

---

## System Architecture
The flowchart below illustrates the packet lifecycle from initial network ingestion down to edge prediction outputs:

```mermaid
flowchart TD
    A["Raw OT/IoT Network Traffic (SCADA, Modbus, DNP3, OPC-UA)"] --> B["CICFlowMeter Feature Extraction (80+ raw features)"]
    B --> C["Data Preprocessing (Normalization, Encoding, Train/Test Split)"]
    C --> D["BWOA Feature Selection (n_agents=30, max_iter=100, V-shaped Transfer Function)"]
    D --> E["Optimal Feature Subset (Reduced Dimensionality)"]
    E --> F["CNN-LSTM Classifier (Conv1D Spatial + LSTM Temporal)"]
    F --> G["Attack Classification (Normal / DoS / Probe / R2L / U2R)"]
    G --> H{"Deployment Target"}
    H --> I["Cloud Deployment (AWS EC2)"]
    H --> J["Edge Deployment (Raspberry Pi, sub-100ms)"]
```

---

## Three-Phase Research Roadmap
The development phases and timelines, showing the milestone presentation in October 2026:

```mermaid
gantt
    title Research Project Roadmap
    dateFormat  YYYY-MM-DD
    section Phase 1
    Data Collection :active, des1, 2026-01-01, 2026-06-30
    section Phase 2
    Model Adaptation : des2, 2026-04-01, 2026-10-31
    section Phase 3
    Edge Deployment : des3, 2026-08-01, 2026-11-30
    section Milestone
    October 2026 Presentation : milestone, m1, 2026-10-12, 1d
```

---

## Pipeline Overview
The modular structures of our data pipeline, model training, and edge evaluations:

```mermaid
flowchart TD
    subgraph Data["Data Ingestion & Loaders"]
        nsl["NSL-KDD Loader"]
        swat["SWaT Loader"]
        bat["BATADAL Loader"]
        ot["OT Collector"]
    end
    subgraph Model["Model & Optimization Pipeline"]
        bwoa["BWOA Optimizer"]
        fit["Fitness Evaluation"]
        cnn_lstm["CNN-LSTM Model"]
        train["Model Trainer"]
    end
    subgraph Eval["Evaluation & Utilities"]
        met["Performance Metrics"]
        edge["Edge Benchmark"]
        logs["Structured Logger"]
        viz["Visualizer"]
    end
    Data --> Model
    Model --> Eval
```

---

## Repository Structure
```text
.
├── .ai/
│   ├── context.md
│   ├── rules.md
│   └── skills.md
├── data/
│   ├── features/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       └── .gitkeep
├── docs/
│   ├── api_reference.md
│   ├── architecture.md
│   ├── bwoa_algorithm.md
│   ├── contribution_guide.md
│   ├── dataset_guide.md
│   ├── experiment_guide.md
│   └── results.md
├── figures/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── notebooks/
│   ├── 01_eda_nslkdd.ipynb
│   ├── 02_bwoa_feature_selection.ipynb
│   ├── 03_cnn_lstm_baseline.ipynb
│   ├── 04_ot_traffic_adaptation.ipynb
│   └── 05_edge_deployment_benchmark.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── batadal.py
│   │   ├── nsl_kdd.py
│   │   ├── ot_collector.py
│   │   └── swat.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── edge_benchmark.py
│   │   └── metrics.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_lstm.py
│   │   └── trainer.py
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── bwoa.py
│   │   └── fitness.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── visualizer.py
├── tests/
│   ├── test_bwoa.py
│   ├── test_cnn_lstm.py
│   └── test_metrics.py
├── config.yaml
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Installation
Clone the repository and install all dependencies:
```bash
git clone https://github.com/mhiskall282/unesco-project.git
cd unesco-project
pip install -r requirements.txt
```

### 2. Set Up Datasets
Place raw datasets in the designated directories:
- NSL-KDD: `data/raw/KDDTrain+.txt` and `data/raw/KDDTest+.txt`
- SWaT: `data/raw/swat.csv`
- BATADAL: `data/raw/batadal.csv`

### 3. Run Experiments
Execute the notebooks sequentially or run unit tests to verify local setup:
```bash
python -m unittest discover -s tests
```

---

## Experiment Results

| Dataset | Accuracy | Precision | Recall | F1 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| NSL-KDD (baseline) | 0.7870 | 0.8231 | 0.7870 | 0.7657 | 68.27ms |
| NSL-KDD + BWOA | 0.6783 | 0.7637 | 0.6783 | 0.7052 | 35.60ms |
| SWaT (adapted) | TBD | TBD | TBD | TBD | TBD |
| Custom OT Dataset | TBD | TBD | TBD | TBD | TBD |

---

## BWOA Feature Selection
The optimization lifecycle runs iteratively through encircling, exploration, and bubble-net search mechanisms:

```mermaid
flowchart TD
    A["Initialize n_agents whale positions (Random binary vectors length n_features)"] --> B["Evaluate fitness for each agent"]
    B --> C["Identify best agent (leader position X*)"]
    C --> D{"For each iteration t"}
    D --> E["Update a: 2 to 0 linearly"]
    E --> F{"Random p < 0.5?"}
    F -->|Yes| G{"|A| < 1?"}
    G -->|Yes bubble-net| H["Shrinking encircling (X = X* - A * D)"]
    G -->|No search| I["Random agent search (Exploration phase)"]
    F -->|No spiral| J["Spiral position update (X = D * e^bl * cos(2pi l) + X*)"]
    H --> K["Apply V-shaped Transfer Function"]
    I --> K
    J --> K
    K --> L["Flip bits probabilistically (Binary position update)"]
    L --> M["Evaluate fitness for updated agents"]
    M --> N{"t < max_iter?"}
    N -->|Yes| D
    N -->|No| O["Return best feature mask and fitness history"]
```

---

## SDG Alignment

| SDG | Goal | How This Project Contributes |
| :--- | :--- | :--- |
| SDG 9 | Industry, Innovation and Infrastructure | Strengthens cybersecurity resilience of digitalizing mining infrastructure. |
| SDG 8 | Decent Work and Economic Growth | Protects worker safety and operational continuity at mining operations. |
| SDG 17 | Partnerships for the Goals | Russian-African collaborative data collection and research pathway. |

---

## Team
- **John Okyere**: Team Lead, AI Security Researcher (University of Education, Winneba and University of Ghana; Co-founder, Kayaba Labs; ICP Ambassador; johnokyere.xyz).
- **[Team Member 2]**: Researcher, SCADA/IIoT Data Acquisition Specialist.
- **[Team Member 3]**: Edge Deployment and Quantization Engineer.

---

## Citation
If you reference this research project in your publications, please cite the work below:

```bibtex
@inproceedings{okyere2026securing,
  author    = {Okyere, John},
  title     = {Securing the Digital Mine: A Metaheuristic Optimized Deep Learning Framework for Intrusion Detection in IoT Enabled Mineral Resource Operations},
  booktitle = {Proceedings of the Russian-African Forum-Contest of Young Scientists: Future Engineers of the World: The Foundation of Sustainable Development},
  publisher = {Empress Catherine II Saint Petersburg Mining University},
  year      = {2026},
  address   = {Saint Petersburg, Russia},
  month     = {October}
}
```

---

## References
1. Mirjalili, S., & Lewis, A. (2016). The whale optimization algorithm. *Advances in Engineering Software*, 95, 51:67. https://doi.org/10.1016/j.advengsoft.2016.01.008
2. Kheddar, H., Himeur, Y., & Awad, A. I. (2023). Deep transfer learning for intrusion detection in industrial control networks. *Journal of Network and Computer Applications*. https://doi.org/10.48550/arXiv.2304.10550
3. Alanazi, M., Mahmood, A., & Chowdhury, M. J. M. (2022). SCADA vulnerabilities and attacks. *Computers & Security*, 125, 103028. https://doi.org/10.1016/j.cose.2022.103028
4. Almomani, O., Akour, I., & Habeb, A. (2025). Cyberattack detection for SCADA in IIoT. *Symmetry*, 17(4), 480. https://doi.org/10.3390/sym17040480
5. Krishnaveni, S., Chen, T. M., Sivamohan, S., & Subbiah, S. (2025). Hybrid metaheuristic IDS for WSN. *Cluster Computing*, 28, 5248. https://doi.org/10.1007/s10586-025-05248-6
6. Anand, M., & Arul, U. (2024). WOA enhanced LSTM for intrusion detection. *Cryptography*, 8(4), 73. https://doi.org/10.3390/cryptography8040073

---

## License
Distributed under the MIT License. See `LICENSE` for more details.
