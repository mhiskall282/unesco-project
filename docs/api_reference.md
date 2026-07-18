# API Reference Documentation

This document provides a programming reference for classes, functions, and modules inside the `src/` folder.

---

## 1. Optimization Package (`src/optimization/`)

### `BinaryWhaleOptimizer`
`from src.optimization.bwoa import BinaryWhaleOptimizer`

A candidate feature search wrapper based on whale hunting mechanics, adapted for discrete spaces.

#### Methods
* `__init__(n_agents: int, n_features: int, max_iter: int, fitness_fn: Callable, b: float = 1.0)`: Initializes search spaces and agent populations.
* `optimize(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[np.ndarray, List[float]]`: Runs search loops and returns (best_mask, fitness_history).
* `_transfer_function(v: np.ndarray) -> np.ndarray`: V-shaped transfer function mapping continuous steps to probability arrays.

---

### `FeatureFitnessEvaluator`
`from src.optimization.fitness import FeatureFitnessEvaluator`

Computes feature selection quality.

#### Methods
* `__init__(alpha: float = 0.88)`: Initializes weight constraints.
* `calculate_fitness(features_mask: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float`: Evaluates sub-features fitness via Decision Trees.

---

## 2. Models Package (`src/models/`)

### `build_cnn_lstm`
`from src.models.cnn_lstm import build_cnn_lstm`

```python
def build_cnn_lstm(
    input_shape: Tuple[int, int],
    n_classes: int,
    filters: int = 64,
    kernel_size: int = 3,
    lstm_units: int = 128,
    dropout_rate: float = 0.3
) -> tf.keras.Model:
```
Assembles Conv1D and LSTM blocks. Returns compiled neural networks.

---

### `ModelTrainer`
`from src.models.trainer import ModelTrainer`

Orchestrates backpropagation optimization runs.

#### Methods
* `__init__(config: Dict[str, Any])`: Configures learning rates and file saving paths.
* `train(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.callbacks.History`: Executes model fits.

---

## 3. Data Loaders (`src/data/`)

### `NSLKDDLoader`
Loads and maps NSL-KDD data structures.

#### Methods
* `load(path: str) -> pd.DataFrame`: Ingests raw data.
* `preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]`: Cleans and encodes classes.
* `train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2)`: Splits data.
* `normalize(X_train: np.ndarray, X_test: np.ndarray)`: Scales values.

---

### `SWaTLoader` and `BATADALLoader`
Loaders for SWaT and BATADAL industrial datasets conforming to the same interface as `NSLKDDLoader`.

---

### `OTTrafficCollector`
Manages packet captures and CICFlowMeter logs.

#### Methods
* `configure(interface: str, output_dir: str, cicflowmeter_path: str)`: Configures paths.
* `capture(duration_seconds: int) -> str`: Captures flows.
* `align_features_to_nslkdd(df: pd.DataFrame) -> pd.DataFrame`: Maps columns to standard baseline shapes.

---

## 4. Evaluation and Benchmarks (`src/evaluation/`)

### `ExperimentMetrics`
Calculates precision, recall, confusion matrix, ROC-AUC, and latency profiles.

#### Methods
* `compute(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]`: Computes metrics.
* `latency_profile(model, X_sample: np.ndarray, n_runs: int = 100) -> Dict[str, float]`: Computes latencies.
* `to_json(metrics_dict: Dict[str, Any], path: str)`: Dumps to JSON files.

---

### `EdgeBenchmark`
Checks edge hardware compatibility and handles quantization.

#### Methods
* `load_model(model_path: str)`: Ingests saved models.
* `benchmark_latency(X_sample: np.ndarray, num_runs: int = 100) -> Dict[str, float]`: Evaluates inference latency.
* `benchmark_memory() -> Dict[str, float]`: Evaluates RAM allocation size.
* `quantize_model(model: tf.keras.Model, quantization_type: str = "float16") -> str`: Creates TFLite files.
* `check_deployment_readiness(latency_dict: Dict[str, float], memory_dict: Dict[str, float]) -> Tuple[bool, str]`: Evaluates readiness.
