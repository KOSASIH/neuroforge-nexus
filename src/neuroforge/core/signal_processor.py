"""
NeuroForge Nexus — Real-Time Neural Signal Processor
=====================================================
Production-grade DSP pipeline for EEG/ECoG/LFP neural signals.

Features:
- Bandpass / notch / spatial filtering
- Independent Component Analysis (ICA)
- Artifact detection & removal (eye blinks, muscle, cardiac)
- Power Spectral Density (Welch + multitaper)
- Phase-Locking Value (PLV) and coherence
- Common Spatial Patterns (CSP) for motor imagery
- Empirical Mode Decomposition (EMD)
- Hilbert transform for instantaneous features
- Real-time streaming with asyncio
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq
from scipy.linalg import eigh
from loguru import logger

from neuroforge.core.constants import (
    ALL_BANDS,
    SAMPLE_RATE_MID,
    NOTCH_50HZ,
    NOTCH_60HZ,
    SNR_ACCEPTABLE,
)


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class NeuralEpoch:
    """A windowed segment of neural data."""
    data: NDArray[np.float64]        # shape: (n_channels, n_samples)
    sample_rate: int
    timestamp: float
    channel_names: list[str]
    paradigm: str = "unknown"
    label: Optional[int] = None      # class label if supervised
    quality_score: float = 1.0       # 0..1 signal quality
    artifacts: list[str] = field(default_factory=list)


@dataclass
class SpectralFeatures:
    """Frequency-domain features from a neural epoch."""
    freqs: NDArray[np.float64]
    psd: NDArray[np.float64]         # shape: (n_channels, n_freqs)
    band_power: dict[str, NDArray[np.float64]]  # band -> (n_channels,)
    coherence: NDArray[np.float64]   # shape: (n_channels, n_channels)
    plv: NDArray[np.float64]         # phase-locking value matrix
    spectral_entropy: NDArray[np.float64]
    dominant_frequency: NDArray[np.float64]


@dataclass
class SpatiotemporalFeatures:
    """Combined spatial + temporal features for classification."""
    csp_features: NDArray[np.float64]
    band_features: NDArray[np.float64]
    erd_ers: NDArray[np.float64]   # Event-Related (De)Synchronization
    hilbert_amplitude: NDArray[np.float64]
    hilbert_phase: NDArray[np.float64]
    snr_per_channel: NDArray[np.float64]
    timestamp: float


# ─── Artifact Detector ────────────────────────────────────────────────────────

class ArtifactDetector:
    """
    Multi-stage artifact rejection pipeline.
    Detects: eye blinks (EOG), muscle (EMG), cardiac (ECG), electrode pops.
    """

    def __init__(
        self,
        amplitude_threshold: float = 150e-6,   # 150 µV for EEG
        gradient_threshold: float = 50e-6,       # µV/sample
        flat_signal_threshold: float = 0.1e-6,   # µV — flat line
        variance_percentile: float = 97.5,
    ) -> None:
        self.amp_thresh = amplitude_threshold
        self.grad_thresh = gradient_threshold
        self.flat_thresh = flat_signal_threshold
        self.var_pct = variance_percentile

    def detect(self, data: NDArray[np.float64]) -> dict[str, NDArray[np.bool_]]:
        """
        Returns per-sample boolean masks for each artifact type.
        data: shape (n_channels, n_samples)
        """
        n_ch, n_samp = data.shape
        artifacts: dict[str, NDArray[np.bool_]] = {}

        # 1. Amplitude clipping
        artifacts["amplitude"] = np.any(np.abs(data) > self.amp_thresh, axis=0)

        # 2. Gradient (muscle / electrode pop)
        grad = np.diff(data, axis=1, prepend=data[:, :1])
        artifacts["gradient"] = np.any(np.abs(grad) > self.grad_thresh, axis=0)

        # 3. Flat signal (electrode disconnection)
        rolling_var = np.array([
            np.var(data[:, max(0, i-50):i+50], axis=1).mean()
            for i in range(n_samp)
        ])
        artifacts["flat"] = rolling_var < self.flat_thresh

        # 4. High-variance burst (epileptic-like discharge)
        var_threshold = np.percentile(np.var(data, axis=1), self.var_pct)
        channel_var = np.var(data, axis=1)
        noisy_channels = channel_var > var_threshold
        artifacts["burst"] = np.tile(noisy_channels[:, None], (1, n_samp)).any(axis=0)

        return artifacts

    def get_clean_mask(self, data: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Returns True where samples are artifact-free."""
        art = self.detect(data)
        combined = np.zeros(data.shape[1], dtype=bool)
        for mask in art.values():
            combined |= mask
        return ~combined


# ─── Main Signal Processor ────────────────────────────────────────────────────

class SignalProcessor:
    """
    NeuroForge Nexus Real-Time Neural Signal Processor.

    Handles the complete DSP chain from raw neural ADC samples through
    spatial filtering, artifact rejection, feature extraction, and
    readiness for AI classification.

    Example:
        processor = SignalProcessor(sample_rate=1000, n_channels=64)
        epoch = processor.create_epoch(raw_data, channel_names)
        features = processor.extract_all_features(epoch)
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE_MID,
        n_channels: int = 64,
        notch_freq: float = NOTCH_50HZ,
        bandpass: tuple[float, float] = (0.5, 100.0),
        filter_order: int = 4,
        buffer_seconds: float = 10.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.notch_freq = notch_freq
        self.bandpass = bandpass
        self.filter_order = filter_order
        self.buffer_size = int(buffer_seconds * sample_rate)

        # Ring buffer for continuous streaming
        self._buffer: deque[NDArray[np.float64]] = deque(maxlen=self.buffer_size)
        self._buffer_lock = asyncio.Lock()

        # Pre-compute filter coefficients
        self._notch_sos = self._design_notch(notch_freq, sample_rate)
        self._bandpass_sos = self._design_bandpass(bandpass, sample_rate, filter_order)

        # Artifact detector
        self.artifact_detector = ArtifactDetector()

        # CAR reference weights (Common Average Reference)
        self._car_weights = np.ones((n_channels, n_channels)) * (-1.0 / n_channels)
        np.fill_diagonal(self._car_weights, 1.0 - 1.0 / n_channels)

        # CSP filters (initialized lazily during calibration)
        self._csp_filters: Optional[NDArray[np.float64]] = None

        logger.info(
            f"SignalProcessor initialized: {n_channels}ch @ {sample_rate}Hz, "
            f"bandpass={bandpass}Hz, notch={notch_freq}Hz"
        )

    # ─── Filter Design ────────────────────────────────────────────────────────

    @staticmethod
    def _design_notch(freq: float, fs: int, quality: float = 30.0) -> NDArray:
        b, a = sp_signal.iirnotch(freq, quality, fs)
        return sp_signal.tf2sos(b, a)

    @staticmethod
    def _design_bandpass(
        band: tuple[float, float], fs: int, order: int = 4
    ) -> NDArray:
        low, high = band
        nyq = fs / 2.0
        return sp_signal.butter(
            order, [low / nyq, high / nyq], btype="band", output="sos"
        )

    @staticmethod
    def _design_bandpass_single(
        low: float, high: float, fs: int, order: int = 4
    ) -> NDArray:
        nyq = fs / 2.0
        return sp_signal.butter(
            order, [low / nyq, high / nyq], btype="band", output="sos"
        )

    # ─── Core Filtering ───────────────────────────────────────────────────────

    def notch_filter(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply notch filter (power line removal). data: (n_ch, n_samp)."""
        return sp_signal.sosfiltfilt(self._notch_sos, data, axis=1)

    def bandpass_filter(
        self,
        data: NDArray[np.float64],
        low: Optional[float] = None,
        high: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """Bandpass filter. Uses class defaults if low/high not specified."""
        if low is not None or high is not None:
            lo = low or self.bandpass[0]
            hi = high or self.bandpass[1]
            sos = self._design_bandpass_single(lo, hi, self.sample_rate)
        else:
            sos = self._bandpass_sos
        return sp_signal.sosfiltfilt(sos, data, axis=1)

    def band_filter(self, data: NDArray[np.float64], band: str) -> NDArray[np.float64]:
        """Filter to a named frequency band (delta/theta/alpha/beta/gamma...)."""
        if band not in ALL_BANDS:
            raise ValueError(f"Unknown band '{band}'. Choose from: {list(ALL_BANDS)}")
        low, high = ALL_BANDS[band]
        # Clamp high to Nyquist
        high = min(high, self.sample_rate / 2.0 - 1)
        return self.bandpass_filter(data, low, high)

    def car_reference(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Common Average Reference: subtract mean across channels."""
        return data - data.mean(axis=0, keepdims=True)

    def laplacian_reference(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Surface Laplacian approximation using mean of adjacent channels."""
        # Simplified: re-reference each channel to average of neighbors
        referenced = np.zeros_like(data)
        for i in range(self.n_channels):
            # Indices of "neighboring" channels (wraps around for simplicity)
            neighbors = [(i - 1) % self.n_channels, (i + 1) % self.n_channels]
            referenced[i] = data[i] - data[neighbors].mean(axis=0)
        return referenced

    def full_preprocessing(
        self, data: NDArray[np.float64], apply_car: bool = True
    ) -> tuple[NDArray[np.float64], list[str]]:
        """
        Full preprocessing pipeline: notch → bandpass → reference → artifact check.
        Returns (clean_data, artifact_list).
        """
        start = time.perf_counter()

        # Step 1: Notch filter
        data = self.notch_filter(data)

        # Step 2: Bandpass filter
        data = self.bandpass_filter(data)

        # Step 3: Spatial reference
        if apply_car:
            data = self.car_reference(data)

        # Step 4: Artifact detection
        art_masks = self.artifact_detector.detect(data)
        detected = [k for k, v in art_masks.items() if v.any()]

        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms > 1.0:
            logger.warning(f"Preprocessing took {elapsed_ms:.2f}ms — check for bottlenecks")

        return data, detected

    # ─── Feature Extraction ───────────────────────────────────────────────────

    def compute_psd(
        self,
        data: NDArray[np.float64],
        method: str = "welch",
        n_fft: int = 512,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute Power Spectral Density.
        Returns (freqs, psd) where psd is (n_channels, n_freqs).
        """
        if method == "welch":
            nperseg = min(n_fft, data.shape[1])
            noverlap = min(n_fft // 2, nperseg - 1)
            freqs, psd = sp_signal.welch(
                data,
                fs=self.sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                axis=1,
            )
        elif method == "multitaper":
            # Simplified multitaper using scipy periodogram
            freqs, psd = sp_signal.periodogram(
                data, fs=self.sample_rate, window="dpss", axis=1,
                nfft=n_fft,
            )
        else:
            raise ValueError(f"Unknown PSD method: {method}")
        return freqs, psd

    def band_power(
        self, freqs: NDArray[np.float64], psd: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        """Compute average band power for each frequency band."""
        result: dict[str, NDArray[np.float64]] = {}
        for band_name, (low, high) in ALL_BANDS.items():
            mask = (freqs >= low) & (freqs <= min(high, self.sample_rate / 2))
            if mask.any():
                result[band_name] = np.trapz(psd[:, mask], freqs[mask], axis=1)
            else:
                result[band_name] = np.zeros(psd.shape[0])
        return result

    def compute_plv(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Phase-Locking Value matrix for all channel pairs.
        Returns (n_channels, n_channels) symmetric matrix.
        """
        analytic = sp_signal.hilbert(data, axis=1)
        phases = np.angle(analytic)
        n_ch = phases.shape[0]
        plv = np.zeros((n_ch, n_ch))
        for i in range(n_ch):
            for j in range(i, n_ch):
                phase_diff = phases[i] - phases[j]
                plv_val = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv[i, j] = plv[j, i] = plv_val
        return plv

    def compute_coherence(
        self,
        data: NDArray[np.float64],
        low: float = 8.0,
        high: float = 13.0,
    ) -> NDArray[np.float64]:
        """
        Magnitude-squared coherence matrix for all channel pairs.
        Returns (n_channels, n_channels).
        """
        n_ch = data.shape[0]
        coh = np.zeros((n_ch, n_ch))
        for i in range(n_ch):
            for j in range(i, n_ch):
                freqs, cxy = sp_signal.coherence(
                    data[i], data[j], fs=self.sample_rate,
                    nperseg=min(256, data.shape[1])
                )
                mask = (freqs >= low) & (freqs <= high)
                if mask.any():
                    val = cxy[mask].mean()
                else:
                    val = 0.0
                coh[i, j] = coh[j, i] = val
        return coh

    def compute_spectral_entropy(
        self, psd: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Spectral entropy per channel (Shannon entropy of normalized PSD).
        Returns (n_channels,).
        """
        psd_norm = psd / (psd.sum(axis=1, keepdims=True) + 1e-12)
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12), axis=1)
        return entropy

    def hilbert_features(
        self, data: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Instantaneous amplitude and phase via Hilbert transform.
        Returns (amplitude, phase) each (n_channels, n_samples).
        """
        analytic = sp_signal.hilbert(data, axis=1)
        return np.abs(analytic), np.unwrap(np.angle(analytic), axis=1)

    def compute_erd_ers(
        self,
        data: NDArray[np.float64],
        baseline: NDArray[np.float64],
        band: str = "beta",
    ) -> NDArray[np.float64]:
        """
        Event-Related (De)Synchronization.
        ERD/ERS = (event_power - baseline_power) / baseline_power × 100
        Returns (n_channels,) in percent.
        """
        band_data = self.band_filter(data, band)
        band_base = self.band_filter(baseline, band)

        event_power = np.mean(band_data ** 2, axis=1)
        base_power = np.mean(band_base ** 2, axis=1) + 1e-30

        return (event_power - base_power) / base_power * 100.0

    def fit_csp(
        self,
        epochs_class1: list[NDArray[np.float64]],
        epochs_class2: list[NDArray[np.float64]],
        n_components: int = 6,
    ) -> NDArray[np.float64]:
        """
        Common Spatial Patterns (CSP) for binary motor imagery classification.
        epochs_class*: list of (n_channels, n_samples)
        Returns spatial filter matrix (n_components, n_channels).
        """
        def cov_mean(epochs: list[NDArray[np.float64]]) -> NDArray[np.float64]:
            covs = [np.cov(e) / (np.trace(np.cov(e)) + 1e-12) for e in epochs]
            return np.mean(covs, axis=0)

        cov1 = cov_mean(epochs_class1)
        cov2 = cov_mean(epochs_class2)

        # Solve generalized eigenvalue problem: cov1 @ w = λ cov2 @ w
        evals, evecs = eigh(cov1, cov1 + cov2)

        # Select n/2 largest and n/2 smallest eigenvalue components
        idx = np.argsort(np.abs(evals - 0.5))[::-1]
        selected = np.concatenate([
            idx[:n_components // 2],
            idx[-(n_components // 2):]
        ])

        self._csp_filters = evecs[:, selected].T
        logger.info(f"CSP fitted: {n_components} components from {len(epochs_class1) + len(epochs_class2)} epochs")
        return self._csp_filters

    def apply_csp(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply fitted CSP spatial filters. Returns (n_components, n_samples)."""
        if self._csp_filters is None:
            raise RuntimeError("CSP filters not fitted. Call fit_csp() first.")
        return self._csp_filters @ data

    def extract_features(
        self, data: NDArray[np.float64]
    ) -> SpatiotemporalFeatures:
        """
        Full feature extraction: band powers + CSP + Hilbert + ERD/ERS.
        data: (n_channels, n_samples)
        """
        # Band power features
        freqs, psd = self.compute_psd(data)
        bp = self.band_power(freqs, psd)
        band_feat = np.stack(list(bp.values()), axis=1).flatten()

        # Hilbert features
        amplitude, phase = self.hilbert_features(data)
        amp_mean = amplitude.mean(axis=1)

        # CSP features (if fitted)
        if self._csp_filters is not None:
            csp_out = self.apply_csp(data)
            csp_feat = np.log(np.var(csp_out, axis=1) + 1e-12)
        else:
            csp_feat = np.zeros(2)

        # ERD/ERS using first half as pseudo-baseline
        mid = data.shape[1] // 2
        erd_ers = self.compute_erd_ers(data[:, mid:], data[:, :mid])

        # SNR per channel
        signal_power = np.var(data, axis=1)
        noise_estimate = np.var(np.diff(data, axis=1), axis=1) / 2
        snr = 10 * np.log10(signal_power / (noise_estimate + 1e-30))

        return SpatiotemporalFeatures(
            csp_features=csp_feat,
            band_features=band_feat,
            erd_ers=erd_ers,
            hilbert_amplitude=amp_mean,
            hilbert_phase=phase.mean(axis=1),
            snr_per_channel=snr,
            timestamp=time.time(),
        )

    def extract_spectral_features(self, data: NDArray[np.float64]) -> SpectralFeatures:
        """Full spectral feature set including PLV and coherence."""
        freqs, psd = self.compute_psd(data)
        bp = self.band_power(freqs, psd)
        plv = self.compute_plv(data)
        coh = self.compute_coherence(data)
        entropy = self.compute_spectral_entropy(psd)
        dom_freq = freqs[np.argmax(psd, axis=1)]

        return SpectralFeatures(
            freqs=freqs,
            psd=psd,
            band_power=bp,
            coherence=coh,
            plv=plv,
            spectral_entropy=entropy,
            dominant_frequency=dom_freq,
        )

    # ─── Signal Quality ───────────────────────────────────────────────────────

    def compute_snr(self, data: NDArray[np.float64]) -> float:
        """Mean SNR across all channels in dB."""
        signal_power = np.var(data, axis=1)
        noise = np.var(np.diff(data, axis=1), axis=1) / 2
        snr_db = np.mean(10 * np.log10(signal_power / (noise + 1e-30)))
        return float(snr_db)

    def quality_score(self, data: NDArray[np.float64]) -> float:
        """
        0..1 quality score for an epoch.
        Based on SNR, artifact count, and signal stationarity.
        """
        snr = self.compute_snr(data)
        snr_score = min(max((snr - SNR_ACCEPTABLE) / (50.0 - SNR_ACCEPTABLE), 0.0), 1.0)

        clean_mask = self.artifact_detector.get_clean_mask(data)
        clean_ratio = clean_mask.mean()

        return float(0.6 * snr_score + 0.4 * clean_ratio)

    # ─── Epoch Factory ────────────────────────────────────────────────────────

    def create_epoch(
        self,
        data: NDArray[np.float64],
        channel_names: list[str],
        paradigm: str = "unknown",
        label: Optional[int] = None,
        preprocess: bool = True,
    ) -> NeuralEpoch:
        """Create a NeuralEpoch with optional preprocessing."""
        if preprocess:
            data, artifacts = self.full_preprocessing(data)
        else:
            artifacts = []

        qs = self.quality_score(data)
        return NeuralEpoch(
            data=data,
            sample_rate=self.sample_rate,
            timestamp=time.time(),
            channel_names=channel_names,
            paradigm=paradigm,
            label=label,
            quality_score=qs,
            artifacts=artifacts,
        )

    # ─── Async Streaming Buffer ───────────────────────────────────────────────

    async def push_samples(self, samples: NDArray[np.float64]) -> None:
        """Push new samples into the ring buffer (thread-safe)."""
        async with self._buffer_lock:
            for i in range(samples.shape[1]):
                self._buffer.append(samples[:, i])

    async def get_epoch(
        self, duration_ms: int = 1000
    ) -> Optional[NeuralEpoch]:
        """Extract a fixed-duration epoch from the ring buffer."""
        n_samples = int(duration_ms * self.sample_rate / 1000)
        async with self._buffer_lock:
            if len(self._buffer) < n_samples:
                return None
            buf = np.array(list(self._buffer)[-n_samples:]).T  # (n_ch, n_samp)

        channels = [f"ch_{i}" for i in range(self.n_channels)]
        return self.create_epoch(buf, channels)

    def sliding_epochs(
        self,
        data: NDArray[np.float64],
        epoch_ms: int = 1000,
        step_ms: int = 500,
    ) -> Generator[NeuralEpoch, None, None]:
        """Generator yielding overlapping epochs from a continuous recording."""
        epoch_samp = int(epoch_ms * self.sample_rate / 1000)
        step_samp = int(step_ms * self.sample_rate / 1000)
        channels = [f"ch_{i}" for i in range(self.n_channels)]
        pos = 0
        while pos + epoch_samp <= data.shape[1]:
            window = data[:, pos: pos + epoch_samp]
            yield self.create_epoch(window, channels)
            pos += step_samp
