# AI Assistant Rules and Instructions

This document lists the strict rules and instructions that the AI assistant must follow when generating code, scripts, documentation, or reports in this repository.

## 1. Code Style and Typing
- **PEP8 Compliance**: All Python code must strictly follow the PEP8 style guide.
- **Type Hints**: All variables, function parameters, and return types must be explicitly typed using standard Python type hints.
- **Google-Style Docstrings**: Every module, class, and function must include complete Google-style docstrings detailing arguments, return types, raises, and descriptions.

## 2. Punctuation Constraint
- **No Em Dashes**: Never use the em dash (—) character in any generated files, documentation, markdown files, or code comments. Use commas, colons, or semicolons instead.

## 3. Academic Citations
- **Verifiable Citations**: When referencing academic work, always provide real, verifiable citations in the format: Author (Year), with a DOI URL where possible. Never fabricate citations.

## 4. Machine Learning Standards
- **Framework**: The default machine learning and deep learning framework is TensorFlow/Keras. Other frameworks (e.g., PyTorch) must not be used unless explicitly requested by the user.
- **Feature Selection**: The Binary Whale Optimization Algorithm (BWOA) implementation in `src/optimization/bwoa.py` must be used for all feature selection experiments.

## 5. Directory Structure and Paths
- **Logs and Checkpoints**: Save all experiment outputs, tensorboard logs, and metric data into the `logs/` directory. Save trained models and weights to the `models/` directory.
- **Dataset Access**: All data paths must reference the `data/` subdirectories:
  - Raw data: `data/raw/`
  - Preprocessed data: `data/processed/`
  - Selected features: `data/features/`

## 6. Edge Deployment Limits
- **Memory Footprint**: Target edge hardware has constrained RAM; code must be optimized to run within a maximum of 1GB RAM.
- **Latency Limit**: Inference latency must be sub-100ms per packet/sample.

## 7. Metrics and Evaluation
- **Evaluation Set**: Every experiment and model evaluation must track and report:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Latency (inference time per sample in milliseconds)

## 8. Academic Writing Style
- **Objective Tone**: Follow rigorous academic writing guidelines in all project papers, reports, or abstracts.
- **Third Person**: Do not use first-person pronouns (I, we, our) in abstracts or introductions.
- **Supported Claims**: Cite a reference for every technical, domain-specific, or architectural claim.
