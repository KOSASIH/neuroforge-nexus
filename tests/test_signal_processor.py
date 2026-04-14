"""
Tests: Signal Processor
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from neuroforge.core.signal_processor import SignalProcessor, ArtifactDetector


@pytest.fixture
def sp():
    return SignalProcessor(sample_rate=256, n_channels=16)


@pytest.fixture
def eeg_data(sp):
    """Synthetic EEG: 2s of 16-channel data."""
    n_samples = 512
    t = np.linspace(0, 2, n_samples)
    data = np.zeros((16, n_samples))
    for ch in range(16):
        data[ch] = (
            10e-6 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)  # alpha
            + 5e-6 * np.sin(2 * np.pi * 20 * t)    # beta
            + 2e-6 * np.random.randn(n_samples)      # noise
        )
    return data


class TestSignalProcessor:
    def test_notch_filter(self, sp, eeg_data):
        filtered = sp.notch_filter(eeg_data)
        assert filtered.shape == eeg_data.shape
        # Power at 50 Hz should be reduced
        from scipy.signal import welch
        f, psd_orig = welch(eeg_data[0], fs=sp.sample_rate, nperseg=128)
        f, psd_filt = welch(filtered[0], fs=sp.sample_rate, nperseg=128)
        idx_50 = np.argmin(np.abs(f - 50))
        # Filtered power should be <= original at 50 Hz
        assert psd_filt[idx_50] <= psd_orig[idx_50] * 1.1  # allow 10% tolerance

    def test_bandpass_filter(self, sp, eeg_data):
        filtered = sp.bandpass_filter(eeg_data, low=8.0, high=30.0)
        assert filtered.shape == eeg_data.shape
        assert not np.allclose(filtered, eeg_data)

    def test_band_filter_alpha(self, sp, eeg_data):
        alpha = sp.band_filter(eeg_data, "alpha")
        assert alpha.shape == eeg_data.shape

    def test_car_reference(self, sp, eeg_data):
        car = sp.car_reference(eeg_data)
        # Mean across channels should be ~0
        assert np.abs(car.mean(axis=0)).max() < 1e-10

    def test_compute_psd(self, sp, eeg_data):
        freqs, psd = sp.compute_psd(eeg_data)
        assert psd.shape == (16, len(freqs))
        assert (psd >= 0).all()
        assert len(freqs) > 0

    def test_band_power(self, sp, eeg_data):
        freqs, psd = sp.compute_psd(eeg_data)
        bp = sp.band_power(freqs, psd)
        assert "alpha" in bp
        assert "beta" in bp
        for band, power in bp.items():
            assert power.shape == (16,)
            assert (power >= 0).all()

    def test_spectral_entropy(self, sp, eeg_data):
        _, psd = sp.compute_psd(eeg_data)
        entropy = sp.compute_spectral_entropy(psd)
        assert entropy.shape == (16,)
        assert (entropy >= 0).all()

    def test_hilbert_features(self, sp, eeg_data):
        amp, phase = sp.hilbert_features(eeg_data)
        assert amp.shape == eeg_data.shape
        assert phase.shape == eeg_data.shape
        assert (amp >= 0).all()

    def test_compute_snr(self, sp, eeg_data):
        snr = sp.compute_snr(eeg_data)
        assert isinstance(snr, float)
        assert np.isfinite(snr)

    def test_quality_score(self, sp, eeg_data):
        qs = sp.quality_score(eeg_data)
        assert 0.0 <= qs <= 1.0

    def test_full_preprocessing(self, sp, eeg_data):
        processed, artifacts = sp.full_preprocessing(eeg_data)
        assert processed.shape == eeg_data.shape
        assert isinstance(artifacts, list)

    def test_extract_features(self, sp, eeg_data):
        features = sp.extract_features(eeg_data)
        assert features.band_features is not None
        assert features.hilbert_amplitude.shape == (16,)
        assert features.erd_ers.shape == (16,)
        assert features.snr_per_channel.shape == (16,)

    def test_sliding_epochs(self, sp, eeg_data):
        epochs = list(sp.sliding_epochs(eeg_data, epoch_ms=500, step_ms=250))
        assert len(epochs) > 0
        for epoch in epochs:
            assert epoch.data.shape[0] == 16

    def test_csp_fit_apply(self, sp, eeg_data):
        class1 = [eeg_data + np.random.randn(*eeg_data.shape) * 1e-7 for _ in range(5)]
        class2 = [eeg_data * 0.5 + np.random.randn(*eeg_data.shape) * 1e-7 for _ in range(5)]
        filters = sp.fit_csp(class1, class2, n_components=4)
        assert filters.shape[0] == 4
        assert filters.shape[1] == 16

        csp_output = sp.apply_csp(eeg_data)
        assert csp_output.shape[0] == 4


class TestArtifactDetector:
    def test_clean_signal(self):
        detector = ArtifactDetector(amplitude_threshold=500e-6)
        data = np.random.randn(16, 256) * 10e-6
        mask = detector.get_clean_mask(data)
        assert mask.dtype == bool
        assert mask.shape == (256,)

    def test_amplitude_artifact_detection(self):
        detector = ArtifactDetector(amplitude_threshold=100e-6)
        data = np.random.randn(16, 256) * 10e-6
        # Inject large amplitude spike
        data[0, 128] = 500e-6  # Above threshold
        art = detector.detect(data)
        assert "amplitude" in art
        assert art["amplitude"][128]
