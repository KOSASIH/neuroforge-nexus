"""Tests: Brain Simulator"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from neuroforge.simulation.brain_simulator import BrainSimulator, pink_noise, eye_blink_artifact


class TestBrainSimulator:
    def setup_method(self):
        self.sim = BrainSimulator(n_channels=16, sample_rate=128)

    def test_resting_state(self):
        session = self.sim.simulate_resting_state(duration_s=5.0)
        assert session.data.shape[0] == 16
        assert session.data.shape[1] == 5 * 128
        assert session.snr_db > 0
        assert len(session.channel_names) == 16

    def test_motor_imagery(self):
        session = self.sim.simulate_motor_imagery(n_trials=10, trial_duration_s=2.0)
        assert session.data.shape[0] == 16
        assert len(session.events) == 10
        assert session.paradigm == "motor_imagery"
        classes = set(e["class"] for e in session.events)
        assert len(classes) >= 1

    def test_p300(self):
        session = self.sim.simulate_p300(n_trials=50)
        assert session.data.shape[0] == 16
        assert len(session.events) == 50
        target_count = sum(1 for e in session.events if e["target"])
        assert target_count > 0

    def test_ssvep(self):
        session = self.sim.simulate_ssvep(n_trials=20, stim_frequencies=[8.0, 12.0])
        assert session.data.shape[0] == 16
        assert session.paradigm == "ssvep"

    def test_pink_noise(self):
        noise = pink_noise(1000, 1.0)
        assert len(noise) == 1000
        assert noise.std() > 0

    def test_eye_blink(self):
        blink = eye_blink_artifact(1000, 128)
        assert len(blink) == 1000
