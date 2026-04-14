"""
NeuroForge Nexus — Neural Signal Constants
==========================================
Canonical constants for BCI signal processing, electrode standards,
frequency bands, and hardware specifications.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Final

# ─── Sampling Rates (Hz) ────────────────────────────────────────────────────
SAMPLE_RATE_LOW: Final[int] = 250       # Consumer-grade EEG
SAMPLE_RATE_MID: Final[int] = 1_000     # Research-grade EEG
SAMPLE_RATE_HIGH: Final[int] = 2_000    # Intra-cortical / ECoG
SAMPLE_RATE_ULTRA: Final[int] = 30_000  # Spike-level neural recording
SAMPLE_RATE_NEXUS: Final[int] = 100_000 # NeuroForge HyperNeural Forge

# ─── EEG Frequency Bands (Hz) ────────────────────────────────────────────────
BAND_DELTA: Final[tuple[float, float]] = (0.5, 4.0)    # Deep sleep
BAND_THETA: Final[tuple[float, float]] = (4.0, 8.0)    # Drowsiness, memory
BAND_ALPHA: Final[tuple[float, float]] = (8.0, 13.0)   # Relaxed wakefulness
BAND_BETA: Final[tuple[float, float]] = (13.0, 30.0)   # Active cognition
BAND_GAMMA: Final[tuple[float, float]] = (30.0, 100.0) # High cognitive load
BAND_HIGH_GAMMA: Final[tuple[float, float]] = (100.0, 300.0) # Cortical spike
BAND_RIPPLE: Final[tuple[float, float]] = (80.0, 120.0) # Memory replay

ALL_BANDS: Final[dict[str, tuple[float, float]]] = {
    "delta": BAND_DELTA,
    "theta": BAND_THETA,
    "alpha": BAND_ALPHA,
    "beta": BAND_BETA,
    "gamma": BAND_GAMMA,
    "high_gamma": BAND_HIGH_GAMMA,
    "ripple": BAND_RIPPLE,
}

# ─── Standard Electrode Systems ───────────────────────────────────────────────
ELECTRODES_10_20 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]

ELECTRODES_64CH = [
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7",
    "FC5", "FC3", "FC1", "C1", "C3", "C5", "T7", "TP7",
    "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9",
    "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz",
    "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4",
    "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCz", "Cz",
    "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
]

ELECTRODES_128CH = ELECTRODES_64CH + [f"EX{i}" for i in range(1, 65)]
ELECTRODES_256CH = ELECTRODES_128CH + [f"AI{i}" for i in range(1, 129)]

# ─── Signal Quality Thresholds ────────────────────────────────────────────────
SNR_ACCEPTABLE: Final[float] = 10.0   # dB — minimum usable signal
SNR_GOOD: Final[float] = 20.0         # dB — research quality
SNR_EXCELLENT: Final[float] = 30.0    # dB — clinical grade
SNR_NEXUS: Final[float] = 50.0        # dB — NeuroForge Nexus target

IMPEDANCE_MAX_KOHM: Final[float] = 20.0   # Max electrode impedance (kΩ)
IMPEDANCE_GOOD_KOHM: Final[float] = 5.0   # Good electrode contact (kΩ)

# ─── Neural Latency Targets ───────────────────────────────────────────────────
LATENCY_CONSUMER_MS: Final[float] = 100.0   # Consumer BCI latency
LATENCY_RESEARCH_MS: Final[float] = 10.0    # Research-grade latency
LATENCY_CLINICAL_MS: Final[float] = 1.0     # Clinical-grade latency
LATENCY_NEXUS_MS: Final[float] = 0.001      # NeuroForge sub-ms target

# ─── Cognitive State Enumeration ─────────────────────────────────────────────
class CognitiveState(Enum):
    DEEP_SLEEP = auto()
    LIGHT_SLEEP = auto()
    DROWSY = auto()
    RELAXED = auto()
    FOCUSED = auto()
    FLOW = auto()
    HYPERFOCUSED = auto()
    STRESSED = auto()
    MEDITATIVE = auto()
    CREATIVE = auto()
    NEXUS_AUGMENTED = auto()  # Full NeuroForge augmentation active


# ─── BCI Paradigms ───────────────────────────────────────────────────────────
class BCIParadigm(Enum):
    MOTOR_IMAGERY = "motor_imagery"          # Imagined movement
    P300 = "p300"                            # Oddball event-related potential
    SSVEP = "ssvep"                          # Steady-state visual evoked
    IMAGINED_SPEECH = "imagined_speech"      # Silent speech decoding
    SLOW_CORTICAL = "slow_cortical"          # Slow cortical potentials
    ERROR_RELATED = "error_related"          # Error-related negativity
    MENTAL_WORKLOAD = "mental_workload"      # Cognitive load monitoring
    EMOTION_REGULATION = "emotion_reg"      # Affective state control
    NEXUS_OMNI = "nexus_omni"               # NeuroForge proprietary protocol


# ─── Neural Modality Flags ───────────────────────────────────────────────────
class NeuralModality(Enum):
    EEG = "eeg"        # Electroencephalography (scalp)
    ECoG = "ecog"      # Electrocorticography (cortical surface)
    LFP = "lfp"        # Local field potential (depth electrode)
    SPIKE = "spike"    # Single-unit action potential
    FNIRS = "fnirs"    # Functional near-infrared spectroscopy
    MEG = "meg"        # Magnetoencephalography
    SEEG = "seeg"      # Stereo-EEG (depth electrodes)
    NEXUS_IMPL = "nexus_impl"  # NeuroForge HyperNeural implant


# ─── Hardware Specs ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HardwareSpec:
    name: str
    channels: int
    sample_rate: int
    resolution_bits: int
    modality: NeuralModality
    wireless: bool
    implantable: bool

SPEC_NEXUS_ALPHA = HardwareSpec(
    name="NeuroForge HyperNeural α",
    channels=4096,
    sample_rate=SAMPLE_RATE_NEXUS,
    resolution_bits=24,
    modality=NeuralModality.NEXUS_IMPL,
    wireless=True,
    implantable=True,
)

SPEC_NEXUS_CONSUMER = HardwareSpec(
    name="NeuroForge Headset Pro",
    channels=64,
    sample_rate=SAMPLE_RATE_MID,
    resolution_bits=24,
    modality=NeuralModality.EEG,
    wireless=True,
    implantable=False,
)

# ─── Signal Processing Window Sizes ──────────────────────────────────────────
EPOCH_DURATION_MS: Final[int] = 1000     # Default epoch length (ms)
OVERLAP_PERCENT: Final[float] = 0.5      # Window overlap (50%)
FEATURE_WINDOW_MS: Final[int] = 500      # Feature extraction window
PREDICTION_HORIZON_MS: Final[int] = 100  # Intent prediction lookahead

# ─── Notch Filter Frequencies (Hz) ───────────────────────────────────────────
NOTCH_50HZ: Final[float] = 50.0    # European power line
NOTCH_60HZ: Final[float] = 60.0    # North American power line
NOTCH_HARMONICS: Final[list[float]] = [100.0, 150.0, 200.0, 250.0]
