"""Tests: AI Engine (Predix OmniMind)"""
import numpy as np
import pytest
import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from neuroforge.ai.predix_omnimind import (
    PredixOmniMind, EEGNet, NeuralTransformer, CognitiveAmplifier,
)
from neuroforge.core.constants import CognitiveState
import torch


@pytest.fixture
def oracle():
    return PredixOmniMind(n_channels=16, n_samples=128, sample_rate=128, model_size="base")


class TestEEGNet:
    def test_forward(self):
        model = EEGNet(n_classes=4, n_channels=16, n_samples=128, sample_rate=128)
        x = torch.randn(2, 1, 16, 128)
        out = model(x)
        assert out.shape == (2, 4)


class TestNeuralTransformer:
    def test_forward(self):
        model = NeuralTransformer(n_features=32, n_classes=4, n_heads=4, n_layers=2, d_model=64)
        x = torch.randn(2, 10, 32)
        out = model(x)
        assert out.shape == (2, 4)


class TestPredixOmniMind:
    def test_predict_intent(self, oracle):
        features = np.random.randn(128)
        result = asyncio.run(oracle.predict_intent(features))
        assert result.label in oracle.MOTOR_LABELS
        assert 0.0 <= result.confidence <= 1.0
        assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=0.01)

    def test_assess_cognitive_state(self, oracle):
        features = np.random.randn(128)
        profile = asyncio.run(oracle.assess_cognitive_state(features))
        assert isinstance(profile.state, CognitiveState)
        assert 0.0 <= profile.attention_index <= 1.0
        assert 0.0 <= profile.flow_probability <= 1.0

    def test_model_stats(self, oracle):
        stats = oracle.get_model_stats()
        assert stats["total_params"] > 0


class TestCognitiveAmplifier:
    def test_session_lifecycle(self, oracle):
        amp = CognitiveAmplifier(oracle)
        session_id = amp.start_session()
        assert session_id is not None
        features = np.random.randn(128)
        cmd = asyncio.run(amp.amplify_step(features, CognitiveState.FLOW))
        assert "feedback_type" in cmd
        summary = amp.get_session_summary()
        assert summary["n_steps"] == 1
