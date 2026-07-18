# System Architecture and Methodology

This document outlines the high-level system architecture and research methodology for "Securing the Digital Mine".

---

## 1. Pipeline Architecture
The system processes raw IoT/OT telemetry in a pipeline, mapping network packets onto compressed feature spaces for deep learning classification. The complete packet lifecycle from network sniffing to localized edge inference:

```mermaid
flowchart TD
    A[Raw OT/IoT Network Traffic\nSCADA, Modbus, DNP3, OPC-UA] --> B[CICFlowMeter\nFeature Extraction\n80+ raw features]
    B --> C[Data Preprocessing\nNormalization, Encoding, Train/Test Split]
    C --> D[BWOA Feature Selection\nn_agents=30, max_iter=100\nV-shaped Transfer Function]
    D --> E[Optimal Feature Subset\nReduced Dimensionality]
    E --> F[CNN-LSTM Classifier\nConv1D Spatial + LSTM Temporal]
    F --> G[Attack Classification\nNormal / DoS / Probe / R2L / U2R]
    G --> H{Deployment Target}
    H --> I[Cloud Deployment\nAWS EC2]
    H --> J[Edge Deployment\nRaspberry Pi, sub-100ms]
```

### Ingestion and Extraction
Raw network streams (SCADA protocol headers, packet structures, payload elements) are captured. CICFlowMeter generates flow-based bidirectional statistics.

### Preprocessing and Normalization
Extracted features are mapped, label-encoded, split, and normalized via Standard Scaler to avoid feature skew.

### Binary Whale Optimization Algorithm (BWOA)
A metaheuristic wrapper runs over candidate feature subsets, searching for an optimal binary feature mask. This operates under a fitness function balancing error rate minimization with maximum dimensionality reduction.

### CNN-LSTM Classifier
A hybrid neural network. Conv1D layers extract spatial correlations from network sequence metrics. LSTM layers model the temporal patterns over packet flow time-steps.

### Target Deployment
The final compressed model is quantized and compiled for local low-power execution (Raspberry Pi devices) or cloud dashboards (AWS EC2).

---

## 2. Research Phases
The research workflow comprises three concurrent phases of development and field testing:

```mermaid
flowchart LR
    subgraph Phase1["Phase 1: Data Collection"]
        P1A[Mining Site OT Infrastructure] --> P1B[AWS EC2 Capture Node]
        P1B --> P1C[CICFlowMeter Processing]
        P1C --> P1D[Labeled OT Dataset]
    end
    subgraph Phase2["Phase 2: Model Adaptation"]
        P2A[NSL-KDD Baseline Model] --> P2B[Transfer Learning]
        P1D --> P2B
        P2B --> P2C[OT-Adapted CNN-LSTM]
        P2C --> P2D[Validation on SWaT/BATADAL]
    end
    subgraph Phase3["Phase 3: Edge Deployment"]
        P2D --> P3A[Model Quantization]
        P3A --> P3B[Edge Hardware Testing]
        P3B --> P3C[Production Deployment]
    end
```

### Phase 1: OT Data Collection
Focuses on capturing traffic logs directly from pilot subsoil plants, generating flow statistics, and building a site-specific labeled custom OT dataset.

### Phase 2: Model Adaptation
Validates baseline models on benchmark databases (NSL-KDD), runs the BWOA wrapper, and implements transfer learning. CNN spatial layers are frozen while LSTM layers are retrained to adapt the system to the unique characteristics of mine sites.

### Phase 3: Edge Deployment
Compiles trained models into quantized TFLite representations. Measures memory allocation limits and latency constraints to prove production readiness on constrained local Raspberry Pi-class devices.
