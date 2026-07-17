"""Training orchestrator module for CNN-LSTM deep learning classifiers.

This module provides classes and functions to train the model, log
metrics, and save checkpoints.
"""

from typing import Dict, Any, Tuple
import numpy as np
import tensorflow as tf


class ModelTrainer:
    """Orchestrates model training, metric collection, and weight checkpointing."""

    def __init__(self, config: Dict[str, Any]):
        """Initializes the trainer with configuration parameters.

        Args:
            config: A dictionary containing training and log paths.
        """
        self.config: Dict[str, Any] = config
        self.epochs: int = config.get("epochs", 100)
        self.batch_size: int = config.get("batch_size", 64)
        self.learning_rate: float = config.get("learning_rate", 0.001)
        self.patience: int = config.get("early_stopping_patience", 10)
        self.save_model_dir: str = config.get("save_model_dir", "models/")

    def train(
        self,
        model: tf.keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tf.keras.callbacks.History:
        """Trains the CNN-LSTM model and returns the history.

        Args:
            model: Instantiated Keras model.
            X_train: Training inputs of shape (n_samples, sequence_length, features).
            y_train: Training target labels of shape (n_samples, n_classes).
            X_val: Validation inputs of shape (n_samples, sequence_length, features).
            y_val: Validation target labels of shape (n_samples, n_classes).

        Returns:
            The training history object containing metrics per epoch.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Compile model
        loss_fn = (
            "categorical_crossentropy"
            if y_train.shape[-1] > 1
            else "binary_crossentropy"
        )
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=["accuracy"],
        )

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{self.save_model_dir}/best_cnn_lstm.h5",
                monitor="val_loss",
                save_best_only=True,
            ),
        ]

        # Train model
        history = model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )

        return history
