"""CNN-LSTM hybrid deep learning classifier.

This module provides functions to construct the hybrid deep learning model
comprising 1D Convolutional layers (for spatial feature extraction) and
LSTM layers (for temporal sequence modeling).
"""

from typing import Tuple
import tensorflow as tf


def build_cnn_lstm(
    input_shape: Tuple[int, int],
    n_classes: int,
    filters: int = 64,
    kernel_size: int = 3,
    lstm_units: int = 128,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """Builds and compiles the CNN-LSTM hybrid neural network model.

    Args:
        input_shape: A tuple representing input shape (sequence_length, features).
        n_classes: Number of prediction targets. Use 2 for binary classification.
        filters: Number of filters for the Conv1D layers.
        kernel_size: Dimension of kernel window size in Conv1D.
        lstm_units: Size of hidden units in LSTM layers.
        dropout_rate: Ratio of node dropouts for regularization.

    Returns:
        A compiled tf.keras.Model instance.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # 1D Convolution for spatial feature extraction
    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        activation="relu",
        padding="same",
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding="same")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Secondary Convolutional block
    x = tf.keras.layers.Conv1D(
        filters=filters * 2,
        kernel_size=kernel_size,
        activation="relu",
        padding="same",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding="same")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # LSTM for temporal sequence learning (unrolled to support TFLite without Select TF Ops)
    x = tf.keras.layers.LSTM(units=lstm_units, return_sequences=False, unroll=True)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Classifier output
    if n_classes == 2 or n_classes == 1:
        # Binary classification
        outputs = tf.keras.layers.Dense(n_classes, activation="sigmoid")(x)
    else:
        # Multi-class classification
        outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_IDS")
    return model
