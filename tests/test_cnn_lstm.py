"""Unit tests for the CNN-LSTM deep learning model architecture.

This module contains test cases to verify the construction of the hybrid model layers,
input shapes, and predictions.
"""

import unittest
import numpy as np
import tensorflow as tf
from src.models.cnn_lstm import build_cnn_lstm


class TestCNNLSTMModel(unittest.TestCase):
    """Tests the Keras architecture assembly and forward propagation."""

    def test_model_construction(self) -> None:
        """Verifies that the compiled model has the expected output layers and shape."""
        input_shape = (10, 41)  # sequence_length = 10, features = 41
        n_classes = 2
        
        model = build_cnn_lstm(
            input_shape=input_shape,
            n_classes=n_classes,
            filters=16,
            kernel_size=3,
            lstm_units=32,
            dropout_rate=0.2
        )
        
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape, (None, 10, 41))
        self.assertEqual(model.output_shape, (None, 2))

    def test_forward_pass(self) -> None:
        """Checks prediction output dimensions for dummy input batches."""
        input_shape = (5, 20)
        n_classes = 5
        
        model = build_cnn_lstm(
            input_shape=input_shape,
            n_classes=n_classes,
            filters=8,
            kernel_size=3,
            lstm_units=16,
            dropout_rate=0.1
        )
        
        dummy_inputs = np.random.rand(4, 5, 20)  # batch size = 4
        predictions = model.predict(dummy_inputs)
        
        self.assertEqual(predictions.shape, (4, 5))
        # Ensure it outputs probability distributions summing to 1 (softmax)
        np.testing.assert_allclose(np.sum(predictions, axis=-1), np.ones(4), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
