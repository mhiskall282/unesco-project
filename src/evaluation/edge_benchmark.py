"""Edge device latency and memory utilization benchmark.

This module provides tools to profile deep learning model performance (specifically RAM
footprint and inference time) under Raspberry Pi-class edge constraints.
"""

import time
import os
import sys
from typing import Dict, Any, Tuple
import numpy as np


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

    def benchmark_latency(self, model, X_sample: np.ndarray, num_runs: int = 100) -> Dict[str, float]:
        """Profiles the mean, 95th percentile, and 99th percentile inference latency.

        Args:
            model: Model instances (TFLite or Keras) supporting predictions.
            X_sample: Input features representing a single packet or sample.
            num_runs: Number of iterations to run the profiling loop.

        Returns:
            A dictionary containing latency statistics.
        """
        latencies = []
        
        # Warmup run
        _ = model.predict(X_sample)

        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model.predict(X_sample)
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

        latencies_arr = np.array(latencies)
        mean_lat = float(np.mean(latencies_arr))
        p95_lat = float(np.percentile(latencies_arr, 95))
        p99_lat = float(np.percentile(latencies_arr, 99))

        return {
            "mean_latency_ms": mean_lat,
            "p95_latency_ms": p95_lat,
            "p99_latency_ms": p99_lat,
            "meets_latency_target": float(mean_lat < self.target_latency_ms)
        }

    def benchmark_memory(self) -> Dict[str, float]:
        """Profiles the resident set size memory of the current execution process.

        Returns:
            A dictionary indicating current memory usage in megabytes.
        """
        import psutil
        process = psutil.Process(os.getpid())
        ram_bytes = process.memory_info().rss
        ram_mb = float(ram_bytes / (1024.0 * 1024.0))

        return {
            "ram_usage_mb": ram_mb,
            "meets_ram_target": float(ram_mb < self.max_ram_mb)
        }
