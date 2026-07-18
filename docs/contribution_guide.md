# Contribution Guidelines

This document details how researchers and developers can contribute to this project.

---

## 1. Branching Strategy
We follow a Git Flow style development model:
* `main`: Stable release branch containing competition ready checkpoints.
* `develop`: Active integration branch for new features, models, and dataset mappings.
* Feature branches: Named `feature/feature_name` or `experiment/experiment_name`. Merges back into `develop` require pull request reviews.

---

## 2. Coding Standards
All proposed changes must strictly respect the guidelines:
* **PEP8 Compliance**: Standard indentation, naming conventions, and code formatting rules.
* **Type Annotations**: Explicit type hinting on parameters and return signatures.
* **Google-style Docstrings**: Class and function docstrings detailing inputs, outputs, and behaviors.
* **Strict Punctuation Constraints**: Em dashes (—) are completely prohibited in code, comments, and markdown documentation. Use colons, commas, or semicolons instead.

---

## 3. Experiment Logging Conventions
Every code trial or hyperparameter search must write results to the logs directory:
1. Initialize `ExperimentLogger(experiment_name)`.
2. Save hyperparameters using `log_hyperparams()`.
3. Record metrics using `log_metrics()`.
4. Call `save_experiment_summary(logs/run_summary)` to write structured JSON files and plain text logs.

This process ensures that all research trials are fully documented and reproducible.

---

## 4. Verification Checklists
Before submitting pull requests:
1. Run syntax verification checks on all modified code files:
   `python -m py_compile [modified_file.py]`
2. Run the test suite to ensure all unit tests pass:
   `python -m unittest discover -s tests`
3. Verify that the notebooks run to completion without errors.
