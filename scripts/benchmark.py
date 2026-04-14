#!/usr/bin/env python3
"""
NeuroForge Nexus — BCI Pipeline Benchmark
==========================================
Benchmarks the full signal processing + AI inference pipeline.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --all
    python scripts/benchmark.py --component signal_processor
"""

import argparse
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from neuroforge.core.signal_processor import SignalProcessor
from neuroforge.ai.predix_omnimind import PredixOmniMind
from neuroforge.quantum.quantum_optimizer import QuantumOptimizer
from neuroforge.simulation.brain_simulator import BrainSimulator
import asyncio
from loguru import logger


def benchmark_signal_processor(n_iter: int = 100) -> dict:
    sp = SignalProcessor(sample_rate=256, n_channels=64)
    data = np.random.randn(64, 256) * 10e-6

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        sp.full_preprocessing(data)
        times.append((time.perf_counter() - t0) * 1000)

    feat_times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        sp.extract_features(data)
        feat_times.append((time.perf_counter() - t0) * 1000)

    return {
        "preprocessing_mean_ms": round(np.mean(times), 3),
        "preprocessing_std_ms": round(np.std(times), 3),
        "preprocessing_max_ms": round(np.max(times), 3),
        "feature_extraction_mean_ms": round(np.mean(feat_times), 3),
        "meets_1ms_latency": np.mean(times) < 1.0,
        "n_iter": n_iter,
    }


def benchmark_ai_inference(n_iter: int = 50) -> dict:
    oracle = PredixOmniMind(n_channels=16, n_samples=128, sample_rate=128)
    features = np.random.randn(128)

    async def run():
        times = []
        for _ in range(n_iter):
            pred = await oracle.predict_intent(features)
            times.append(pred.latency_ms)
        return times

    times = asyncio.run(run())

    return {
        "inference_mean_ms": round(np.mean(times), 3),
        "inference_p95_ms": round(np.percentile(times, 95), 3),
        "inference_p99_ms": round(np.percentile(times, 99), 3),
        "throughput_fps": round(1000 / np.mean(times), 1),
        "n_iter": n_iter,
    }


def benchmark_quantum_optimizer(n_iter: int = 5) -> dict:
    optimizer = QuantumOptimizer(n_qubits=6, depth=2)
    matrix = np.random.randn(6, 6)
    matrix = (matrix + matrix.T) / 2  # Symmetric

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        optimizer.optimize_neural_weights(matrix, method="qaoa")
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "qaoa_mean_ms": round(np.mean(times), 2),
        "n_qubits": optimizer.n_qubits,
        "n_iter": n_iter,
    }


def benchmark_brain_simulator(n_iter: int = 5) -> dict:
    sim = BrainSimulator(n_channels=32, sample_rate=128)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        sim.simulate_motor_imagery(n_trials=20)
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "simulation_mean_ms": round(np.mean(times), 2),
        "n_iter": n_iter,
    }


def main():
    parser = argparse.ArgumentParser(description="NeuroForge Nexus Benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--component", choices=["signal", "ai", "quantum", "sim"])
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    results = {}

    if args.all or args.component == "signal" or not args.component:
        logger.info("Benchmarking Signal Processor...")
        results["signal_processor"] = benchmark_signal_processor(args.iters)

    if args.all or args.component == "ai":
        logger.info("Benchmarking AI Inference...")
        results["ai_inference"] = benchmark_ai_inference(min(args.iters, 50))

    if args.all or args.component == "quantum":
        logger.info("Benchmarking Quantum Optimizer...")
        results["quantum_optimizer"] = benchmark_quantum_optimizer(5)

    if args.all or args.component == "sim":
        logger.info("Benchmarking Brain Simulator...")
        results["brain_simulator"] = benchmark_brain_simulator(5)

    # Print results
    print("\n" + "═" * 60)
    print("  NeuroForge Nexus — Benchmark Results")
    print("═" * 60)
    for component, metrics in results.items():
        print(f"\n  📊 {component.upper()}")
        for k, v in metrics.items():
            icon = "✅" if isinstance(v, bool) and v else "❌" if isinstance(v, bool) else "  "
            print(f"     {icon} {k}: {v}")
    print("\n" + "═" * 60)

    return results


if __name__ == "__main__":
    main()
