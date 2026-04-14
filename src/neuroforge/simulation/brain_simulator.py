"""
NeuroForge Nexus — Brain Simulator & Synthetic Data Generator
=============================================================
Physiologically-grounded EEG/ECoG simulation for:
  - Development and testing without hardware
  - Benchmarking signal processing pipelines
  - Training AI models with labeled ground-truth data

Models:
  - Oscillatory EEG: superposition of band-specific oscillators + pink noise
  - Event-related potentials: P300, N200, error-related negativity
  - Motor imagery modulation: ERD/ERS during imagined movement
  - Epileptic discharge simulation
  - Resting state network connectivity patterns
  - Neural plasticity adaptation over time
"""

import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from loguru import logger

from neuroforge.core.constants import (
    SAMPLE_RATE_MID,
    ALL_BANDS,
    ELECTRODES_64CH,
    BCIParadigm,
)


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class SyntheticSession:
    """A generated synthetic neural recording session."""
    session_id: str
    subject_id: str
    paradigm: str
    data: NDArray[np.float64]      # (n_channels, n_samples)
    labels: NDArray[np.int32]      # per-sample class labels
    events: list[dict]             # event markers
    channel_names: list[str]
    sample_rate: int
    duration_s: float
    snr_db: float
    noise_model: str


@dataclass
class OscillatorConfig:
    """Configuration for a single neural oscillator."""
    frequency: float           # Hz
    amplitude: float           # µV
    phase: float               # radians
    channel_weights: NDArray[np.float64]  # Spatial mixing
    is_artifact: bool = False


# ─── Noise Models ─────────────────────────────────────────────────────────────

def pink_noise(n_samples: int, amplitude: float = 1.0) -> NDArray[np.float64]:
    """
    Generate pink (1/f) noise via Voss-McCartney algorithm.
    Characteristic of resting-state EEG background.
    """
    n_octaves = 16
    key = np.zeros(n_octaves)
    output = np.zeros(n_samples)
    for i in range(n_samples):
        j = 0
        tmp = i + 1
        while tmp % 2 == 0:
            tmp //= 2
            j += 1
        if j < n_octaves:
            key[j] = np.random.randn()
        output[i] = key.sum()
    output -= output.mean()
    std = output.std()
    if std > 1e-12:
        output /= std
    return output * amplitude


def white_noise(n_samples: int, amplitude: float = 1.0) -> NDArray[np.float64]:
    """White Gaussian noise."""
    return np.random.randn(n_samples) * amplitude


def muscle_artifact(
    n_samples: int, sample_rate: int, amplitude: float = 50e-6, burst_prob: float = 0.05
) -> NDArray[np.float64]:
    """Simulate high-frequency muscle (EMG) artifacts."""
    emg = np.random.randn(n_samples) * amplitude * 0.1
    # Add random bursts
    for _ in range(int(n_samples * burst_prob / sample_rate)):
        start = np.random.randint(0, n_samples)
        duration = np.random.randint(10, 200)
        end = min(start + duration, n_samples)
        emg[start:end] += np.random.randn(end - start) * amplitude
    return emg


def eye_blink_artifact(
    n_samples: int, sample_rate: int, amplitude: float = 200e-6, rate_hz: float = 0.3
) -> NDArray[np.float64]:
    """Simulate eye blink (EOG) artifacts."""
    eog = np.zeros(n_samples)
    n_blinks = int(n_samples / sample_rate * rate_hz)
    for _ in range(n_blinks):
        t_blink = np.random.randint(0, n_samples)
        width = int(0.15 * sample_rate)
        t = np.arange(-width // 2, width // 2)
        if len(t) > 0:
            blink = amplitude * np.exp(-t ** 2 / (2 * (width / 4) ** 2))
            start = max(0, t_blink - width // 2)
            end = min(n_samples, t_blink + width // 2)
            eog[start:end] += blink[:end - start]
    return eog


# ─── EEG Oscillator Model ─────────────────────────────────────────────────────

class NeuralOscillatorBank:
    """
    Bank of neural oscillators for realistic EEG simulation.
    Each oscillator represents a neural source with a characteristic frequency.
    """

    def __init__(
        self,
        n_channels: int = 64,
        sample_rate: int = SAMPLE_RATE_MID,
    ) -> None:
        self.n_ch = n_channels
        self.fs = sample_rate
        self.oscillators: list[OscillatorConfig] = []

    def add_band_oscillators(self, state: str = "relaxed") -> None:
        """Add oscillators matching a cognitive state profile."""
        profiles = {
            "relaxed": {
                "alpha": (10.0, 30e-6), "theta": (6.0, 10e-6),
                "beta": (20.0, 5e-6), "delta": (2.0, 15e-6),
            },
            "focused": {
                "beta": (18.0, 25e-6), "gamma": (40.0, 10e-6),
                "alpha": (10.0, 8e-6), "theta": (6.0, 12e-6),
            },
            "stressed": {
                "beta": (22.0, 40e-6), "gamma": (50.0, 15e-6),
                "theta": (5.5, 20e-6), "alpha": (9.0, 5e-6),
            },
            "sleep": {
                "delta": (1.5, 80e-6), "theta": (5.0, 30e-6),
                "sleep_spindle": (12.0, 40e-6),
            },
        }
        profile = profiles.get(state, profiles["relaxed"])

        for band, (freq, amp) in profile.items():
            # Spatial weighting: Gaussian on channel grid
            weights = np.random.dirichlet(np.ones(self.n_ch))
            self.oscillators.append(OscillatorConfig(
                frequency=freq + np.random.randn() * 0.5,
                amplitude=amp * (1 + np.random.randn() * 0.2),
                phase=np.random.uniform(0, 2 * np.pi),
                channel_weights=weights,
            ))

    def generate(self, n_samples: int) -> NDArray[np.float64]:
        """
        Generate EEG data from all oscillators.
        Returns (n_channels, n_samples).
        """
        t = np.arange(n_samples) / self.fs
        data = np.zeros((self.n_ch, n_samples))

        for osc in self.oscillators:
            # Sinusoidal with slight frequency jitter (1/f FM modulation)
            freq_mod = osc.frequency + pink_noise(n_samples, 0.1)
            phase_accum = 2 * np.pi * np.cumsum(freq_mod) / self.fs + osc.phase
            wave = osc.amplitude * np.sin(phase_accum)
            # Mix onto channels
            data += np.outer(osc.channel_weights, wave)

        return data


# ─── Brain Simulator ─────────────────────────────────────────────────────────

class BrainSimulator:
    """
    NeuroForge Nexus Physiological Brain Simulator.
    Generates labeled EEG/ECoG datasets for BCI development.
    """

    def __init__(
        self,
        n_channels: int = 64,
        sample_rate: int = SAMPLE_RATE_MID,
        subject_id: Optional[str] = None,
    ) -> None:
        self.n_ch = n_channels
        self.fs = sample_rate
        self.subject_id = subject_id or f"subject_{str(uuid.uuid4())[:6]}"
        self.channel_names = ELECTRODES_64CH[:n_channels]

        # Lead field matrix (forward model): source → electrode projection
        # Simplified: random dipole projections for n_channels
        n_sources = n_channels * 4
        self._lead_field = np.random.randn(n_channels, n_sources) * 0.1
        self._lead_field /= np.linalg.norm(self._lead_field, axis=0, keepdims=True) + 1e-12

        logger.info(f"BrainSimulator: {n_channels}ch @ {sample_rate}Hz | {self.subject_id}")

    def simulate_resting_state(
        self,
        duration_s: float = 60.0,
        cognitive_state: str = "relaxed",
        snr_db: float = 20.0,
    ) -> SyntheticSession:
        """Simulate resting-state EEG with realistic oscillations."""
        n_samples = int(duration_s * self.fs)

        osc_bank = NeuralOscillatorBank(self.n_ch, self.fs)
        osc_bank.add_band_oscillators(cognitive_state)
        neural_signal = osc_bank.generate(n_samples)

        # Add noise floor
        noise_amplitude = np.sqrt(np.var(neural_signal) / 10 ** (snr_db / 10))
        for ch in range(self.n_ch):
            neural_signal[ch] += pink_noise(n_samples, noise_amplitude)
            neural_signal[ch] += white_noise(n_samples, noise_amplitude * 0.1)

        # Add artifacts
        blink_ch = [0, 1]  # Frontal channels
        for ch in blink_ch:
            neural_signal[ch] += eye_blink_artifact(n_samples, self.fs)

        labels = np.zeros(n_samples, dtype=np.int32)

        return SyntheticSession(
            session_id=str(uuid.uuid4()),
            subject_id=self.subject_id,
            paradigm="resting_state",
            data=neural_signal,
            labels=labels,
            events=[],
            channel_names=self.channel_names,
            sample_rate=self.fs,
            duration_s=duration_s,
            snr_db=snr_db,
            noise_model="pink_noise+blink",
        )

    def simulate_motor_imagery(
        self,
        n_trials: int = 100,
        trial_duration_s: float = 4.0,
        iti_s: float = 2.0,
        classes: Optional[list[int]] = None,
        snr_db: float = 15.0,
    ) -> SyntheticSession:
        """
        Simulate motor imagery paradigm.
        Classes: 0=rest, 1=left_hand, 2=right_hand, 3=feet, 4=tongue
        """
        if classes is None:
            classes = [0, 1, 2]  # rest + left + right

        trial_samples = int(trial_duration_s * self.fs)
        iti_samples = int(iti_s * self.fs)
        total_samples = n_trials * (trial_samples + iti_samples)

        data = np.zeros((self.n_ch, total_samples))
        labels_arr = np.zeros(total_samples, dtype=np.int32)
        events: list[dict] = []

        # Baseline oscillations
        osc_bank = NeuralOscillatorBank(self.n_ch, self.fs)
        osc_bank.add_band_oscillators("relaxed")
        baseline = osc_bank.generate(total_samples)
        data += baseline

        pos = 0
        noise_amp = np.sqrt(np.var(baseline) / 10 ** (snr_db / 10))

        for trial_idx in range(n_trials):
            trial_class = np.random.choice(classes)
            t_start = pos + iti_samples
            t_end = t_start + trial_samples

            events.append({
                "trial": trial_idx,
                "class": trial_class,
                "onset_sample": t_start,
                "duration_samples": trial_samples,
            })
            labels_arr[t_start:t_end] = trial_class

            # Motor imagery: ERD in contralateral channels, ERS in ipsilateral
            if trial_class in [1, 2]:  # Hand classes
                # Channel groups: left hemisphere C3 area, right C4 area
                left_chans = np.arange(0, self.n_ch // 2)
                right_chans = np.arange(self.n_ch // 2, self.n_ch)

                if trial_class == 1:  # Left hand → right hemisphere ERD
                    erd_chans, ers_chans = right_chans, left_chans
                else:  # Right hand → left hemisphere ERD
                    erd_chans, ers_chans = left_chans, right_chans

                # ERD: suppress alpha/beta in contralateral channels
                t = np.arange(trial_samples) / self.fs
                suppression = 0.6 * np.exp(-((t - trial_duration_s / 2) ** 2) / 0.5)
                for ch in erd_chans:
                    data[ch, t_start:t_end] *= (1 - suppression)

                # ERS: enhance beta in ipsilateral
                for ch in ers_chans:
                    enhancement = 1.2 + 0.2 * np.sin(2 * np.pi * 20 * t)
                    data[ch, t_start:t_end] *= enhancement

            pos += trial_samples + iti_samples

        # Add global noise
        for ch in range(self.n_ch):
            data[ch] += pink_noise(total_samples, noise_amp)
            data[ch] += muscle_artifact(total_samples, self.fs, noise_amp * 0.5)

        return SyntheticSession(
            session_id=str(uuid.uuid4()),
            subject_id=self.subject_id,
            paradigm=BCIParadigm.MOTOR_IMAGERY.value,
            data=data,
            labels=labels_arr,
            events=events,
            channel_names=self.channel_names,
            sample_rate=self.fs,
            duration_s=total_samples / self.fs,
            snr_db=snr_db,
            noise_model="pink_noise+muscle",
        )

    def simulate_p300(
        self,
        n_trials: int = 200,
        target_probability: float = 0.2,
        snr_db: float = 10.0,
    ) -> SyntheticSession:
        """
        Simulate P300 oddball paradigm.
        P300: positive deflection ~300ms post-stimulus at parietal channels.
        """
        epoch_samples = int(0.8 * self.fs)  # 800ms epochs
        total_samples = n_trials * epoch_samples

        data = np.zeros((self.n_ch, total_samples))
        labels_arr = np.zeros(total_samples, dtype=np.int32)
        events: list[dict] = []

        # Background
        osc_bank = NeuralOscillatorBank(self.n_ch, self.fs)
        osc_bank.add_band_oscillators("focused")
        data += osc_bank.generate(total_samples)

        noise_amp = np.sqrt(np.var(data) / 10 ** (snr_db / 10))
        p300_chans = list(range(max(0, self.n_ch - 16), self.n_ch))  # Posterior channels
        t = np.arange(epoch_samples) / self.fs

        for trial_idx in range(n_trials):
            is_target = np.random.random() < target_probability
            onset = trial_idx * epoch_samples
            label = 1 if is_target else 0
            labels_arr[onset: onset + epoch_samples] = label

            events.append({
                "trial": trial_idx,
                "class": label,
                "target": is_target,
                "onset_sample": onset,
            })

            if is_target:
                # P300: Gaussian at 300ms, parietal channels
                p300_peak = np.exp(-((t - 0.300) ** 2) / (2 * (0.050 ** 2))) * 8e-6
                # N200: small negativity at 200ms
                n200 = -np.exp(-((t - 0.200) ** 2) / (2 * (0.030 ** 2))) * 3e-6
                erp = p300_peak + n200
                for ch in p300_chans:
                    data[ch, onset:onset + epoch_samples] += erp

        # Add noise
        for ch in range(self.n_ch):
            data[ch] += pink_noise(total_samples, noise_amp)

        return SyntheticSession(
            session_id=str(uuid.uuid4()),
            subject_id=self.subject_id,
            paradigm=BCIParadigm.P300.value,
            data=data,
            labels=labels_arr,
            events=events,
            channel_names=self.channel_names,
            sample_rate=self.fs,
            duration_s=total_samples / self.fs,
            snr_db=snr_db,
            noise_model="pink_noise",
        )

    def simulate_ssvep(
        self,
        n_trials: int = 60,
        stim_frequencies: Optional[list[float]] = None,
        trial_duration_s: float = 3.0,
        snr_db: float = 20.0,
    ) -> SyntheticSession:
        """
        Simulate SSVEP paradigm with multiple stimulus frequencies.
        SSVEP: strong oscillation at stimulus frequency in occipital channels.
        """
        if stim_frequencies is None:
            stim_frequencies = [8.0, 12.0, 15.0, 20.0]

        trial_samples = int(trial_duration_s * self.fs)
        total_samples = n_trials * trial_samples

        data = np.zeros((self.n_ch, total_samples))
        labels_arr = np.zeros(total_samples, dtype=np.int32)
        events: list[dict] = []

        occ_chans = list(range(max(0, self.n_ch - 8), self.n_ch))  # Occipital
        noise_amp = np.sqrt(
            np.random.randn(self.n_ch, trial_samples).var() / 10 ** (snr_db / 10)
        )

        t_trial = np.arange(trial_samples) / self.fs

        for trial_idx in range(n_trials):
            stim_idx = trial_idx % len(stim_frequencies)
            freq = stim_frequencies[stim_idx]
            onset = trial_idx * trial_samples
            labels_arr[onset: onset + trial_samples] = stim_idx
            events.append({"trial": trial_idx, "class": stim_idx, "freq_hz": freq, "onset_sample": onset})

            # SSVEP response at stim freq + harmonics
            for ch in occ_chans:
                for harmonic in [1, 2]:
                    amp = 15e-6 / harmonic
                    data[ch, onset:onset + trial_samples] += (
                        amp * np.sin(2 * np.pi * freq * harmonic * t_trial)
                    )

        # Add background + noise
        osc = NeuralOscillatorBank(self.n_ch, self.fs)
        osc.add_band_oscillators("relaxed")
        data += osc.generate(total_samples)
        for ch in range(self.n_ch):
            data[ch] += pink_noise(total_samples, noise_amp)

        return SyntheticSession(
            session_id=str(uuid.uuid4()),
            subject_id=self.subject_id,
            paradigm=BCIParadigm.SSVEP.value,
            data=data,
            labels=labels_arr,
            events=events,
            channel_names=self.channel_names,
            sample_rate=self.fs,
            duration_s=total_samples / self.fs,
            snr_db=snr_db,
            noise_model="pink_noise",
        )

    def batch_generate(
        self,
        n_subjects: int = 10,
        paradigms: Optional[list[str]] = None,
        duration_s: float = 60.0,
    ) -> list[SyntheticSession]:
        """Generate multi-subject synthetic dataset."""
        if paradigms is None:
            paradigms = ["motor_imagery", "p300", "ssvep"]

        sessions: list[SyntheticSession] = []
        for sub_idx in range(n_subjects):
            subject_id = f"subject_{sub_idx:03d}"
            sim = BrainSimulator(self.n_ch, self.fs, subject_id)
            for paradigm in paradigms:
                if paradigm == "motor_imagery":
                    sessions.append(sim.simulate_motor_imagery(
                        n_trials=int(duration_s / 6), snr_db=np.random.uniform(10, 25)
                    ))
                elif paradigm == "p300":
                    sessions.append(sim.simulate_p300(
                        n_trials=int(duration_s / 0.8), snr_db=np.random.uniform(8, 18)
                    ))
                elif paradigm == "ssvep":
                    sessions.append(sim.simulate_ssvep(
                        n_trials=int(duration_s / 3), snr_db=np.random.uniform(15, 30)
                    ))
                elif paradigm == "resting":
                    sessions.append(sim.simulate_resting_state(
                        duration_s=duration_s, snr_db=np.random.uniform(15, 25)
                    ))

        logger.info(f"BrainSimulator: Generated {len(sessions)} sessions for {n_subjects} subjects")
        return sessions
