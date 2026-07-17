# AI Skills and Reference Patterns

This document provides reference implementations, code structures, and technical workflow guidelines for key modules in this repository.

## 1. Binary Whale Optimization Algorithm (BWOA) Patterns
BWOA is a metaheuristic feature selection algorithm. The continuous search space is mapped to a binary space where each feature is represented as 1 (selected) or 0 (not selected).

### Position Representation
Whale positions are represented as binary vectors of length equal to the number of features:
$$\mathbf{X}_i = [x_{i,1}, x_{i,2}, \dots, x_{i,D}], \quad x_{i,j} \in \{0, 1\}$$

### Transfer Function
A V-shaped transfer function maps continuous velocity values to probabilities, which are then used to update the binary states:
$$T(v) = \left| \frac{v}{\sqrt{1 + v^2}} \right|$$
The position is updated by comparing $T(v)$ to a random number $r \in [0, 1]$:
$$x_{i,j}(t+1) = \begin{cases} 1 - x_{i,j}(t) & \text{if } r < T(v) \\ x_{i,j}(t) & \text{otherwise} \end{cases}$$

### Fitness Function
The fitness function balances classification performance against feature reduction:
$$\text{Fitness} = \alpha \times (1 - \text{Accuracy}) + (1 - \alpha) \times \frac{\text{Selected Features}}{\text{Total Features}}$$
Here, $\alpha$ is a user-defined weight (typically 0.88 to prioritize classification accuracy).

---

## 2. CNN-LSTM Architecture Patterns
For multi-class and binary intrusion detection, the combination of CNN and LSTM networks captures both spatial and temporal features of network traffic packet sequences.

### Network Configuration
- **Input Shape**: `(batch_size, sequence_length, feature_dimension)`
- **Spatial Layer**: 1D Convolutional layers (`Conv1D`) extract local feature relationships from the sequences.
- **Pooling Layer**: `MaxPooling1D` reduces spatial dimensions.
- **Temporal Layer**: `LSTM` layers process sequence dynamics and capture long-term dependencies.
- **Dense Output**: A Dense layer with Softmax (multi-class) or Sigmoid (binary) activation.

---

## 3. CICFlowMeter Feature Extraction Workflow
CICFlowMeter generates bidirectional network flow statistics. The process flow is:
1. Capture raw PCAP traffic from network interfaces or read PCAP files.
2. Calculate bidirectional packet metrics including packet lengths, inter-arrival times, flow duration, and flag counts.
3. Output the extracted data as a structured CSV file with over 80 traffic feature columns.

---

## 4. Dataset Loading and Preprocessing Pipelines

### NSL-KDD
1. Read `KDDTrain+.txt` and `KDDTest+.txt`.
2. Clean column names and assign label mappings (e.g., normal vs specific attack types).
3. Apply One-Hot Encoding to categorical features: `protocol_type`, `service`, and `flag`.
4. Apply MinMax scaling to numeric variables.

### SWaT and BATADAL
- SWaT features continuous water flow readings and state variables. Preprocessing involves stripping timestamps, handling missing sensory telemetry, and aligning labels with the normal/attack windows.
- BATADAL features actuator and tank levels under cyber-physical attacks. Preprocessing centers on signal alignment, scaling, and binary attack classification labeling.

---

## 5. Edge Deployment Evaluation
To ensure sub-100ms latency and a footprint under 1GB RAM on edge devices:
- **Model Quantization**: Convert trained TensorFlow models to TensorFlow Lite (TFLite) using float16 or dynamic range post-training quantization.
- **Inference Benchmarking**: Run latency timers on Raspberry Pi-class devices, computing the 95th and 99th percentile inference latencies.
- **Memory Profiling**: Monitor memory allocation using tools like `tracemalloc` or standard OS utilities during model inference execution.

---

## 6. Python Project Structure Conventions
The project maintains a structured package layout to separate optimization, modeling, loading, evaluation, and utilities:
```text
src/
  data/          # Data loaders and capture pipelines
  optimization/  # Metaheuristic optimization modules
  models/        # Deep learning classifier definitions
  evaluation/    # Performance and edge hardware benchmarks
  utils/         # Loggers and visualizers
```
All imports must be absolute: `from src.optimization.bwoa import BinaryWhaleOptimizer`.
