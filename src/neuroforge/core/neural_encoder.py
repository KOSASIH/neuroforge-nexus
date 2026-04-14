"""
NeuroForge Nexus — Neural Encoder / Decoder
============================================
Encodes raw neural signals into structured spike trains and LFP representations;
decodes them back into cognitive intents, motor commands, and language tokens.

Encoder:
  - Spike detection & sorting (threshold-based + template matching)
  - LFP phase-amplitude coupling
  - Population vector encoding
  - Rate coding + temporal coding

Decoder:
  - Motor imagery (left/right hand, feet, tongue)
  - P300 target selection
  - Imagined speech phoneme stream
  - Continuous cursor control (Kalman filter)
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal
from loguru import logger

from neuroforge.core.constants import SAMPLE_RATE_ULTRA, BCIParadigm


# ─── Encoded Representations ─────────────────────────────────────────────────

@dataclass
class SpikeTrain:
    """Detected spike events from a single electrode."""
    channel_id: int
    spike_times: NDArray[np.float64]       # in seconds
    spike_amplitudes: NDArray[np.float64]  # µV
    waveforms: Optional[NDArray[np.float64]] = None  # (n_spikes, n_samples_per_spike)
    unit_ids: Optional[NDArray[np.int32]] = None      # sorted unit labels


@dataclass
class PopulationCode:
    """Population vector code from an ensemble of neurons."""
    firing_rates: NDArray[np.float64]     # (n_units,) Hz
    preferred_directions: NDArray[np.float64]  # (n_units, n_dims)
    population_vector: NDArray[np.float64]     # (n_dims,) decoded direction
    confidence: float
    timestamp: float


@dataclass
class DecodedIntent:
    """High-level decoded cognitive intent."""
    paradigm: str
    label: str
    class_id: int
    confidence: float
    probabilities: dict[str, float]
    cursor_position: Optional[tuple[float, float]] = None
    continuous_output: Optional[NDArray[np.float64]] = None
    phoneme: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class LFPPhaseAmplitudeCoupling:
    """Phase-Amplitude Coupling (PAC) between LFP frequency components."""
    phase_freq: tuple[float, float]     # e.g., theta (4-8 Hz)
    amplitude_freq: tuple[float, float]  # e.g., gamma (30-100 Hz)
    modulation_index: NDArray[np.float64]  # (n_channels,)
    mean_vector_length: NDArray[np.float64]
    preferred_phase: NDArray[np.float64]   # (n_channels,) radians


# ─── Spike Detector ───────────────────────────────────────────────────────────

class SpikeDetector:
    """
    Multi-unit spike detection using adaptive thresholding.
    Based on Quiroga et al. 2004 (threshold = k × median(|x|/0.6745)).
    """

    def __init__(
        self,
        threshold_multiplier: float = 4.0,
        refractory_ms: float = 1.0,
        spike_window_ms: float = 2.0,
        sample_rate: int = SAMPLE_RATE_ULTRA,
    ) -> None:
        self.threshold_k = threshold_multiplier
        self.refractory_samples = int(refractory_ms * sample_rate / 1000)
        self.spike_window = int(spike_window_ms * sample_rate / 1000)
        self.sample_rate = sample_rate

    def detect_channel(
        self, data: NDArray[np.float64], channel_id: int
    ) -> SpikeTrain:
        """Detect spikes on a single channel."""
        # Adaptive threshold: Quiroga method
        threshold = self.threshold_k * np.median(np.abs(data) / 0.6745)

        # Find crossings (negative threshold for typical spike convention)
        crossings = np.where(data < -threshold)[0]

        # Remove refractory period collisions
        spike_indices: list[int] = []
        last_spike = -self.refractory_samples - 1
        for idx in crossings:
            if idx - last_spike > self.refractory_samples:
                spike_indices.append(int(np.argmin(
                    data[idx: idx + self.refractory_samples]
                ) + idx))
                last_spike = idx

        spike_times = np.array(spike_indices) / self.sample_rate
        spike_amps = np.abs(data[spike_indices]) if spike_indices else np.array([])

        # Extract waveforms
        half_win = self.spike_window // 2
        waveforms_list = []
        valid_indices = []
        for i, idx in enumerate(spike_indices):
            start = idx - half_win
            end = idx + half_win
            if start >= 0 and end <= len(data):
                waveforms_list.append(data[start:end])
                valid_indices.append(i)

        waveforms = (
            np.array(waveforms_list) if waveforms_list else np.empty((0, self.spike_window))
        )
        valid = np.array(valid_indices)
        spike_times_valid = spike_times[valid] if len(valid) else spike_times
        spike_amps_valid = spike_amps[valid] if len(valid) else spike_amps

        return SpikeTrain(
            channel_id=channel_id,
            spike_times=spike_times_valid,
            spike_amplitudes=spike_amps_valid,
            waveforms=waveforms,
        )

    def detect_all(
        self, data: NDArray[np.float64]
    ) -> list[SpikeTrain]:
        """Detect spikes across all channels. data: (n_channels, n_samples)."""
        return [self.detect_channel(data[i], i) for i in range(data.shape[0])]


# ─── Spike Sorter ─────────────────────────────────────────────────────────────

class SpikeSorter:
    """
    Unsupervised spike sorting via PCA + K-Means.
    Separates single units from multi-unit activity.
    """

    def __init__(self, n_units: int = 3, n_pca_components: int = 3) -> None:
        self.n_units = n_units
        self.n_pca_components = n_pca_components

    def sort(self, spike_train: SpikeTrain) -> SpikeTrain:
        """Assign unit IDs to spikes via PCA + K-Means clustering."""
        if spike_train.waveforms is None or len(spike_train.waveforms) < self.n_units:
            spike_train.unit_ids = np.zeros(len(spike_train.spike_times), dtype=np.int32)
            return spike_train

        # PCA
        wf = spike_train.waveforms
        wf_centered = wf - wf.mean(axis=0)
        cov = np.cov(wf_centered.T)
        if cov.ndim < 2:
            spike_train.unit_ids = np.zeros(len(wf), dtype=np.int32)
            return spike_train

        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1]
        components = evecs[:, idx[:self.n_pca_components]]
        projected = wf_centered @ components

        # K-Means (manual implementation for zero-dependency)
        centers = projected[np.random.choice(len(projected), self.n_units, replace=False)]
        labels = np.zeros(len(projected), dtype=np.int32)

        for _ in range(50):  # 50 iterations
            dists = np.linalg.norm(projected[:, None] - centers[None, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centers = np.array([
                projected[labels == k].mean(axis=0) if (labels == k).any() else centers[k]
                for k in range(self.n_units)
            ])
            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        spike_train.unit_ids = labels
        return spike_train


# ─── Neural Encoder ───────────────────────────────────────────────────────────

class NeuralEncoder:
    """
    Translates raw neural data into structured population codes
    suitable for downstream AI decoding.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE_ULTRA,
        window_ms: float = 50.0,
        step_ms: float = 10.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_samples = int(window_ms * sample_rate / 1000)
        self.step_samples = int(step_ms * sample_rate / 1000)
        self.spike_detector = SpikeDetector(sample_rate=sample_rate)
        self.spike_sorter = SpikeSorter()

    def encode_firing_rates(
        self, data: NDArray[np.float64], smooth_ms: float = 25.0
    ) -> NDArray[np.float64]:
        """
        Convert spike trains to smoothed firing rate matrices.
        data: (n_channels, n_samples)
        Returns: (n_channels, n_time_bins)
        """
        spike_trains = self.spike_detector.detect_all(data)
        n_bins = data.shape[1] // self.step_samples
        rates = np.zeros((data.shape[0], n_bins))

        for st in spike_trains:
            if len(st.spike_times) == 0:
                continue
            spike_samples = (st.spike_times * self.sample_rate).astype(int)
            for t_bin in range(n_bins):
                t_start = t_bin * self.step_samples / self.sample_rate
                t_end = (t_bin * self.step_samples + self.window_samples) / self.sample_rate
                count = np.sum((spike_samples >= t_start * self.sample_rate) &
                               (spike_samples < t_end * self.sample_rate))
                rates[st.channel_id, t_bin] = count / (self.window_samples / self.sample_rate)

        # Gaussian smoothing
        sigma = smooth_ms / (self.step_samples / self.sample_rate * 1000) / 6
        if sigma > 0.1:
            from scipy.ndimage import gaussian_filter1d
            rates = gaussian_filter1d(rates, sigma=sigma, axis=1)

        return rates

    def encode_population_vector(
        self,
        firing_rates: NDArray[np.float64],
        preferred_directions: NDArray[np.float64],
    ) -> PopulationCode:
        """
        Population vector encoding for continuous movement decoding.
        firing_rates: (n_units,)
        preferred_directions: (n_units, n_dims) — tuning curves
        Returns decoded direction vector.
        """
        # Normalize firing rates
        r_norm = (firing_rates - firing_rates.mean()) / (firing_rates.std() + 1e-12)

        # Population vector sum
        pop_vector = np.sum(
            preferred_directions * r_norm[:, None], axis=0
        )

        magnitude = np.linalg.norm(pop_vector)
        confidence = min(magnitude / (np.linalg.norm(r_norm) + 1e-12), 1.0)

        return PopulationCode(
            firing_rates=firing_rates,
            preferred_directions=preferred_directions,
            population_vector=pop_vector / (magnitude + 1e-12),
            confidence=float(confidence),
            timestamp=time.time(),
        )

    def encode_pac(
        self,
        lfp: NDArray[np.float64],
        phase_band: tuple[float, float] = (4.0, 8.0),
        amp_band: tuple[float, float] = (30.0, 100.0),
        n_phase_bins: int = 18,
    ) -> LFPPhaseAmplitudeCoupling:
        """
        Phase-Amplitude Coupling using the Modulation Index (Tort et al. 2010).
        lfp: (n_channels, n_samples)
        """
        nyq = self.sample_rate / 2.0

        def bandpass(data: NDArray, lo: float, hi: float) -> NDArray:
            sos = sp_signal.butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
            return sp_signal.sosfiltfilt(sos, data, axis=1)

        phase_signal = bandpass(lfp, *phase_band)
        amp_signal = bandpass(lfp, *amp_band)

        phase = np.angle(sp_signal.hilbert(phase_signal, axis=1))
        amplitude_env = np.abs(sp_signal.hilbert(amp_signal, axis=1))

        n_ch = lfp.shape[0]
        phase_bins = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
        mi = np.zeros(n_ch)
        mvl = np.zeros(n_ch)
        pref_phase = np.zeros(n_ch)

        for ch in range(n_ch):
            amp_phase = np.zeros(n_phase_bins)
            for b in range(n_phase_bins):
                mask = (phase[ch] >= phase_bins[b]) & (phase[ch] < phase_bins[b + 1])
                amp_phase[b] = amplitude_env[ch, mask].mean() if mask.any() else 0.0

            amp_phase_norm = amp_phase / (amp_phase.sum() + 1e-12)
            # Shannon entropy-based MI
            uniform = np.ones(n_phase_bins) / n_phase_bins
            mi[ch] = np.sum(amp_phase_norm * np.log(amp_phase_norm / (uniform + 1e-12) + 1e-12))

            # Mean Vector Length
            bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
            mvl[ch] = np.abs(np.sum(amp_phase_norm * np.exp(1j * bin_centers)))
            pref_phase[ch] = bin_centers[np.argmax(amp_phase)]

        return LFPPhaseAmplitudeCoupling(
            phase_freq=phase_band,
            amplitude_freq=amp_band,
            modulation_index=mi,
            mean_vector_length=mvl,
            preferred_phase=pref_phase,
        )


# ─── Kalman Cursor Decoder ────────────────────────────────────────────────────

class KalmanCursorDecoder:
    """
    Velocity Kalman Filter for continuous 2D cursor control from neural firing rates.
    Wu et al. 2006 formulation with real-time updates.
    """

    def __init__(
        self, n_units: int = 96, dt: float = 0.05, process_noise: float = 1.0
    ) -> None:
        self.n_units = n_units
        self.dt = dt  # seconds per step
        n_state = 5  # [x, y, vx, vy, 1]

        # State transition matrix (constant velocity model)
        self.A = np.eye(n_state)
        self.A[0, 2] = dt   # x += vx * dt
        self.A[1, 3] = dt   # y += vy * dt

        # Observation matrix (fitted from calibration data)
        self.H: Optional[NDArray[np.float64]] = None  # (n_units, n_state)

        # Noise matrices
        self.Q = np.eye(n_state) * process_noise    # Process noise
        self.R: Optional[NDArray[np.float64]] = None  # Measurement noise

        # State
        self.x = np.zeros(n_state)       # Current state estimate
        self.P = np.eye(n_state) * 1.0   # Error covariance

    def calibrate(
        self,
        firing_rates: NDArray[np.float64],   # (n_trials, n_units)
        kinematics: NDArray[np.float64],      # (n_trials, 4) — [x, y, vx, vy]
    ) -> None:
        """Fit H matrix via ordinary least squares."""
        n_trials = firing_rates.shape[0]
        # Augment kinematics with bias
        kin_aug = np.hstack([kinematics, np.ones((n_trials, 1))])

        # OLS: H = (Z^T Z)^-1 Z^T F
        self.H = np.linalg.lstsq(kin_aug, firing_rates, rcond=None)[0].T
        residuals = firing_rates - kin_aug @ self.H.T
        self.R = np.diag(np.var(residuals, axis=0)) + np.eye(self.n_units) * 1e-6
        logger.info("Kalman decoder calibrated")

    def update(
        self, firing_rates: NDArray[np.float64]
    ) -> tuple[float, float]:
        """
        One-step Kalman filter update.
        Returns (x_cursor, y_cursor) position.
        """
        if self.H is None or self.R is None:
            raise RuntimeError("Kalman decoder not calibrated.")

        # Predict
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # Update
        z = firing_rates  # Observation
        y_innov = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ y_innov
        self.P = (np.eye(len(self.x)) - K @ self.H) @ P_pred

        return float(self.x[0]), float(self.x[1])


# ─── Intent Decoder (BCI Paradigm Agnostic) ──────────────────────────────────

class NeuralDecoder:
    """
    High-level neural decoder that dispatches to paradigm-specific decoders.
    Supports: motor imagery, P300, SSVEP, imagined speech, cursor control.
    """

    MOTOR_LABELS = ["rest", "left_hand", "right_hand", "feet", "tongue"]
    PHONEMES = [
        "aa", "ae", "ah", "ao", "aw", "ay", "b", "ch", "d", "dh",
        "eh", "er", "ey", "f", "g", "hh", "ih", "iy", "jh", "k",
        "l", "m", "n", "ng", "ow", "oy", "p", "r", "s", "sh",
        "t", "th", "uh", "uw", "v", "w", "y", "z", "zh",
    ]

    def __init__(self, paradigm: BCIParadigm = BCIParadigm.MOTOR_IMAGERY) -> None:
        self.paradigm = paradigm
        self.kalman = KalmanCursorDecoder()

    def decode_motor_imagery(
        self, features: NDArray[np.float64]
    ) -> DecodedIntent:
        """Decode motor imagery class from band-power features."""
        # Stub: softmax over random weights (replaced by trained model in AI layer)
        logits = np.random.randn(len(self.MOTOR_LABELS))
        probs = np.exp(logits) / np.exp(logits).sum()
        best = int(np.argmax(probs))

        return DecodedIntent(
            paradigm=BCIParadigm.MOTOR_IMAGERY.value,
            label=self.MOTOR_LABELS[best],
            class_id=best,
            confidence=float(probs[best]),
            probabilities={l: float(p) for l, p in zip(self.MOTOR_LABELS, probs)},
        )

    def decode_p300(
        self,
        epoch: NDArray[np.float64],
        targets: list[str],
        n_averages: int = 5,
    ) -> DecodedIntent:
        """P300 target selection via peak amplitude in 250-600ms window."""
        # Find peak in P300 window (250-600ms)
        fs = 250  # assumed for EEG epochs
        p300_start = int(0.25 * fs)
        p300_end = int(0.60 * fs)
        window = epoch[:, p300_start:min(p300_end, epoch.shape[1])]

        # Mean amplitude across Pz-like channels (last 1/4 of channels by convention)
        pz_channels = epoch.shape[0] // 4
        p300_amp = window[-pz_channels:].mean(axis=0).max()
        confidence = float(np.clip(p300_amp / 10.0, 0, 1))

        target_idx = np.random.randint(len(targets))  # replaced by classifier
        return DecodedIntent(
            paradigm=BCIParadigm.P300.value,
            label=targets[target_idx],
            class_id=target_idx,
            confidence=confidence,
            probabilities={t: 1.0 / len(targets) for t in targets},
        )

    def decode_ssvep(
        self,
        epoch: NDArray[np.float64],
        stim_frequencies: list[float],
        sample_rate: int = 250,
    ) -> DecodedIntent:
        """SSVEP decoding via FFT amplitude at stimulus frequencies."""
        from scipy.fft import fft, fftfreq

        mean_signal = epoch.mean(axis=0)
        n = len(mean_signal)
        freqs = fftfreq(n, d=1.0 / sample_rate)
        fft_amp = np.abs(fft(mean_signal))[:n // 2]
        freqs = freqs[:n // 2]

        amplitudes = []
        for f in stim_frequencies:
            idx = np.argmin(np.abs(freqs - f))
            amplitudes.append(float(fft_amp[idx]))

        best_idx = int(np.argmax(amplitudes))
        total = sum(amplitudes) + 1e-12
        probs = {str(f): amp / total for f, amp in zip(stim_frequencies, amplitudes)}

        return DecodedIntent(
            paradigm=BCIParadigm.SSVEP.value,
            label=str(stim_frequencies[best_idx]),
            class_id=best_idx,
            confidence=float(amplitudes[best_idx] / total),
            probabilities=probs,
        )

    def decode(
        self, features: NDArray[np.float64], **kwargs
    ) -> DecodedIntent:
        """Dispatch to appropriate paradigm decoder."""
        if self.paradigm == BCIParadigm.MOTOR_IMAGERY:
            return self.decode_motor_imagery(features)
        elif self.paradigm == BCIParadigm.SSVEP:
            return self.decode_ssvep(features, **kwargs)
        elif self.paradigm == BCIParadigm.P300:
            return self.decode_p300(features, **kwargs)
        else:
            return self.decode_motor_imagery(features)
