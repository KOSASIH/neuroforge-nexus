#!/usr/bin/env python3
"""Quick validation script - tests core modules without full test suite overhead."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import numpy as np

print("=" * 60)
print("  NeuroForge Nexus — Core Validation")
print("=" * 60)

# 1. Signal Processor
print("\n[1/5] Signal Processor...")
from neuroforge.core.signal_processor import SignalProcessor
sp = SignalProcessor(sample_rate=256, n_channels=16)
data = np.random.randn(16, 512) * 10e-6
# Add some alpha oscillations
t = np.arange(512) / 256
for ch in range(16):
    data[ch] += 20e-6 * np.sin(2*np.pi*10*t)

processed, artifacts = sp.full_preprocessing(data)
snr = sp.compute_snr(processed)
qs = sp.quality_score(processed)
features = sp.extract_features(processed)
print(f"   ✅ Preprocessing OK | SNR={snr:.1f}dB | Quality={qs:.3f}")
print(f"   ✅ Feature extraction: band_features={features.band_features.shape}")

# 2. Neural Encoder
print("\n[2/5] Neural Encoder...")
from neuroforge.core.neural_encoder import NeuralEncoder, NeuralDecoder
from neuroforge.core.constants import BCIParadigm
encoder = NeuralEncoder(sample_rate=1000)
decoder = NeuralDecoder(BCIParadigm.MOTOR_IMAGERY)
intent = decoder.decode(features.band_features)
print(f"   ✅ Decoded intent: {intent.label} (conf={intent.confidence:.3f})")

# 3. AI Engine
print("\n[3/5] Predix OmniMind AI Engine...")
import asyncio
from neuroforge.ai.predix_omnimind import PredixOmniMind, CognitiveAmplifier
from neuroforge.core.constants import CognitiveState

oracle = PredixOmniMind(n_channels=16, n_samples=128, sample_rate=128)
feat_vec = np.random.randn(128)
prediction = asyncio.run(oracle.predict_intent(feat_vec))
profile = asyncio.run(oracle.assess_cognitive_state(feat_vec))
stats = oracle.get_model_stats()
print(f"   ✅ Prediction: {prediction.label} (conf={prediction.confidence:.3f}, latency={prediction.latency_ms:.1f}ms)")
print(f"   ✅ Cognitive state: {profile.state.name} | flow={profile.flow_probability:.3f}")
print(f"   ✅ Model: {stats['total_params']:,} parameters")

# 4. Quantum Optimizer
print("\n[4/5] Quantum Optimizer...")
from neuroforge.quantum.quantum_optimizer import QuantumOptimizer
optimizer = QuantumOptimizer(n_qubits=6, depth=2)
matrix = np.random.randn(6, 6)
matrix = (matrix + matrix.T) / 2
result = optimizer.optimize_neural_weights(matrix, method="annealing")
rand_bits = optimizer.quantum_random_sample(64)
print(f"   ✅ Annealing: optimal_val={result.optimal_value:.4f}, t={result.runtime_ms:.0f}ms")
print(f"   ✅ Quantum random bits: {len(rand_bits)} bits generated")

# 5. Brain Simulator
print("\n[5/5] Brain Simulator...")
from neuroforge.simulation.brain_simulator import BrainSimulator
sim = BrainSimulator(n_channels=16, sample_rate=128)
mi_session = sim.simulate_motor_imagery(n_trials=10, trial_duration_s=2.0)
p300_session = sim.simulate_p300(n_trials=30)
print(f"   ✅ Motor imagery: {mi_session.data.shape}, {len(mi_session.events)} trials")
print(f"   ✅ P300: {p300_session.data.shape}, {len(p300_session.events)} stimuli")

print("\n" + "=" * 60)
print("  ✅ ALL MODULES VALIDATED SUCCESSFULLY")
print("=" * 60)
