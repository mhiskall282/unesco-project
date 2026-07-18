# Experiment Reproduction Guide

This guide details step-by-step instructions to set up the environment, prepare raw data files, execute the feature selection/training scripts, and replicate our research results.

---

## Step 1: Environment Setup
Ensure Python 3.11 or higher is installed. Build a clean virtual environment and install dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

---

## Step 2: Data Preparation
Obtain the required datasets and place them under raw data directories:

1. **NSL-KDD**:
   The loader automatically attempts downloads. If offline, place `KDDTrain+.txt` and `KDDTest+.txt` into `data/raw/`.
2. **SWaT**:
   Request files from SUTD Singapore and place `swat.csv` into `data/raw/`.
3. **BATADAL**:
   Download challenge files and place `batadal.csv` into `data/raw/`.

---

## Step 3: Running BWOA Feature Selection
To find the optimal feature subsets, run notebook `02_bwoa_feature_selection.ipynb` or execute custom optimization scripts:

```python
from src.data.nsl_kdd import NSLKDDLoader
from src.optimization.bwoa import BinaryWhaleOptimizer
from src.optimization.fitness import FeatureFitnessEvaluator

loader = NSLKDDLoader()
df = loader.load("data/raw/KDDTrain+.txt")
X, y = loader.preprocess(df)

# Use a stratified 3000-sample subset for fitness evaluations
from sklearn.model_selection import train_test_split
_, X_subset, _, y_subset = train_test_split(X, y, test_size=3000, stratify=y, random_state=42)

# v3: alpha=0.3 (70% accuracy weight), min_accuracy=0.75 floor, min_features=10
evaluator = FeatureFitnessEvaluator(alpha=0.3, min_accuracy=0.75, min_features=10)
optimizer = BinaryWhaleOptimizer(
    n_agents=30,
    n_features=X_subset.shape[1],
    max_iter=100,
    fitness_fn=evaluator.calculate_fitness,
    minimum_features=10
)
# patience=15 triggers early stopping when fitness stops improving
best_mask, history = optimizer.optimize(X_subset, y_subset, X_subset, y_subset, patience=15)
```

The resulting feature mask is saved to `data/features/nslkdd_bwoa_mask.npy`.

---

## Step 4: Training the CNN-LSTM Baseline Model
Train the deep learning model on the masked feature subset:

1. Load the binary mask from `data/features/nslkdd_bwoa_mask.npy`.
2. Apply the mask to features and reshape the input to 3D sequences: `(samples, 1, features)`.
3. Call `build_cnn_lstm()` to compile the classifier.
4. Call `ModelTrainer.train()` to run backpropagation.

Outputs (accuracy, loss curves) and checkpoints are written to `figures/` and `models/` folders.

---

## Step 5: Running OT Traffic Domain Adaptation
For transfer learning adaptations to OT mining networks:

1. Capture live or simulated flow logs and map columns via `align_features_to_nslkdd()`.
2. Load the pre-trained baseline classifier model.
3. Freeze the trainable weights of the spatial extractor blocks (Conv1D and BatchNormalization).
4. Train the LSTM sequence layer and Dense output layers on target site flow statistics.

---

## Step 6: Edge Deployment Benchmarking
To profile model memory footprint and inference latency on local targets:

1. Load the saved keras checkpoint.
2. Initialize `EdgeBenchmark` and measure prediction response times.
3. Convert Keras weights to quantized `.tflite` format.
4. Execute `EdgeBenchmark.check_deployment_readiness()` to evaluate edge criteria.
