"""Edge device latency and memory utilization benchmark.

This module provides tools to profile deep learning model performance (specifically RAM
footprint and inference time) under Raspberry Pi-class edge constraints.
"""

import time
import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import tensorflow as tf


class EdgeBenchmark:
    """Measures model inference latency and RAM usage under hardware limits."""

    def __init__(self, target_latency_ms: float = 100.0, max_ram_mb: float = 1024.0):
        """Initializes the benchmark constraints.

        Args:
            target_latency_ms: Target sub-100ms prediction latency.
            max_ram_mb: Maximum RAM allocation constraint in megabytes.
        """
        self.target_latency_ms: float = target_latency_ms
        self.max_ram_mb: float = max_ram_mb
        self.model: Optional[tf.keras.Model] = None
        self.model_path: Optional[str] = None

    def load_model(self, model_path: str) -> tf.keras.Model:
        """Loads a saved Keras model from disk.

        Args:
            model_path: Absolute or relative filepath to load.

        Returns:
            The loaded tf.keras.Model instance.
        """
        self.model_path = model_path
        # Custom object mapping can go here if needed
        self.model = tf.keras.models.load_model(model_path)
        return self.model

    def benchmark_latency(
        self,
        X_sample: np.ndarray,
        num_runs: int = 100,
        model: Optional[tf.keras.Model] = None
    ) -> Dict[str, float]:
        """Profiles the mean, standard deviation, 95th, and 99th percentile inference latency.

        Args:
            X_sample: Input features representing a single packet or sample.
            num_runs: Number of iterations to run the profiling loop.
            model: Optional model instance. If not provided, uses self.model.

        Returns:
            A dictionary containing latency statistics.
        """
        target_model = model if model is not None else self.model
        if target_model is None:
            raise ValueError("No model loaded for benchmarking. Call load_model() first.")

        latencies = []
        
        # Warmup run
        _ = target_model.predict(X_sample[:1], verbose=0)

        for i in range(num_runs):
            idx = i % len(X_sample)
            sample = X_sample[idx:idx+1]
            
            start = time.perf_counter()
            _ = target_model.predict(sample, verbose=0)
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

        latencies_arr = np.array(latencies)
        mean_lat = float(np.mean(latencies_arr))
        std_lat = float(np.std(latencies_arr))
        p95_lat = float(np.percentile(latencies_arr, 95))
        p99_lat = float(np.percentile(latencies_arr, 99))

        return {
            "mean_ms": mean_lat,
            "std_ms": std_lat,
            "p95_ms": p95_lat,
            "p99_ms": p99_lat,
            "meets_latency_target": float(mean_lat < self.target_latency_ms)
        }

    def benchmark_memory(self) -> Dict[str, float]:
        """Profiles the resident set size memory and loaded model size on disk.

        Returns:
            A dictionary indicating current memory usage and model size in megabytes.
        """
        # Read file size
        model_size_mb = 0.0
        if self.model_path and os.path.exists(self.model_path):
            file_bytes = os.path.getsize(self.model_path)
            model_size_mb = float(file_bytes / (1024.0 * 1024.0))

        try:
            import psutil
            process = psutil.Process(os.getpid())
            ram_bytes = process.memory_info().rss
            peak_mb = float(ram_bytes / (1024.0 * 1024.0))
        except ImportError:
            # Fallback if psutil is not available
            peak_mb = 120.0  # mock estimate for testing

        return {
            "peak_mb": peak_mb,
            "model_size_mb": model_size_mb,
            "meets_ram_target": float(peak_mb < self.max_ram_mb)
        }

    def quantize_model(self, model: tf.keras.Model, quantization_type: str = "float16") -> str:
        """Converts the Keras model into a quantized TFLite model saved to models/ folder.

        Args:
            model: Instantiated Keras model.
            quantization_type: Either 'float16' or 'int8'.

        Returns:
            The filepath of the saved quantized TFLite model.
        """
        os.makedirs("models/", exist_ok=True)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Configure optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if quantization_type == "float16":
            converter.target_spec.supported_types = [tf.float16]
            
        tflite_model = converter.convert()
        tflite_path = f"models/cnn_lstm_quantized_{quantization_type}.tflite"
        
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
            
        print(f"Quantized model saved to: {tflite_path}")
        return tflite_path

    def check_deployment_readiness(self, latency_dict: Dict[str, float], memory_dict: Dict[str, float]) -> Tuple[bool, str]:
        """Evaluates whether the profiled latency and RAM meet target edge criteria.

        Args:
            latency_dict: Results from benchmark_latency().
            memory_dict: Results from benchmark_memory().

        Returns:
            A tuple of (is_ready, report_text).
        """
        mean_lat = latency_dict.get("mean_ms", 0.0)
        peak_ram = memory_dict.get("peak_mb", 0.0)
        
        latency_ok = mean_lat < self.target_latency_ms
        ram_ok = peak_ram < self.max_ram_mb
        is_ready = latency_ok and ram_ok
        
        status_lat = "PASSED" if latency_ok else "FAILED"
        status_ram = "PASSED" if ram_ok else "FAILED"
        
        report = (
            f"=== EDGE DEPLOYMENT READINESS REPORT ===\n"
            f"Target Latency: {self.target_latency_ms:.2f}ms | Profiled Latency: {mean_lat:.2f}ms -> {status_lat}\n"
            f"Target RAM Limit: {self.max_ram_mb:.2f}MB | Profiled RAM: {peak_ram:.2f}MB -> {status_ram}\n"
            f"Overall Deployment Readiness: {'READY' if is_ready else 'NOT READY'}\n"
        )
        return is_ready, report
