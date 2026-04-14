"""
NeuroForge Nexus — Predix OmniMind AI Engine
=============================================
Sentient AI oracle for real-time neural intent prediction,
cognitive state assessment, and adaptive cognitive amplification.

Architecture:
  - EEGNet: Compact depthwise CNN optimized for EEG classification
  - NeuralTransformer: Multi-head attention on temporal neural features
  - OmniMindOracle: Ensemble predictor with Bayesian uncertainty estimation
  - CognitiveAmplifier: State-driven neurofeedback amplification engine

All models use PyTorch with full async inference support.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from loguru import logger

from neuroforge.core.constants import CognitiveState, BCIParadigm


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class NeuralPrediction:
    """AI prediction result with uncertainty estimation."""
    label: str
    class_id: int
    confidence: float
    uncertainty: float           # Epistemic uncertainty (model)
    aleatoric: float             # Aleatoric uncertainty (data noise)
    probabilities: dict[str, float]
    attention_weights: Optional[NDArray[np.float64]] = None
    latency_ms: float = 0.0
    model_version: str = "OmniMind-1.0"
    timestamp: float = field(default_factory=time.time)


@dataclass
class CognitiveProfile:
    """Real-time cognitive state profile."""
    state: CognitiveState
    attention_index: float     # 0..1
    workload_index: float      # 0..1
    stress_level: float        # 0..1
    flow_probability: float    # 0..1
    creativity_index: float    # 0..1
    memory_load: float         # 0..1
    emotional_valence: float   # -1 (negative) .. +1 (positive)
    arousal: float             # 0..1
    iq_amplification: float    # 0..1 — active cognitive boost factor
    timestamp: float = field(default_factory=time.time)


@dataclass
class CognitiveAmplificationSession:
    """Neurofeedback amplification session results."""
    session_id: str
    duration_s: float
    target_state: CognitiveState
    achieved_state: CognitiveState
    peak_attention: float
    peak_flow: float
    neurofeedback_commands: list[dict]
    performance_gain: float    # % improvement
    timestamp: float = field(default_factory=time.time)


# ─── EEGNet Architecture ─────────────────────────────────────────────────────

class EEGNet(nn.Module):
    """
    EEGNet: Compact EEG-specific CNN (Lawhern et al. 2018).
    Depthwise separable convolutions optimized for BCI classification.

    Input: (batch, 1, n_channels, n_samples)
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 64,
        n_samples: int = 256,
        sample_rate: int = 128,
        F1: int = 8,      # Temporal filters
        D: int = 2,       # Depth multiplier
        F2: int = 16,     # Pointwise filters
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        T = n_samples

        # Block 1: Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, sample_rate // 2), padding=(0, sample_rate // 4), bias=False),
            nn.BatchNorm2d(F1),
        )

        # Block 2: Depthwise spatial convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(F1, D * F1, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Block 3: Separable convolution
        kernel_size = 16
        self.separable_conv = nn.Sequential(
            nn.Conv2d(D * F1, F2, (1, kernel_size), padding=(0, kernel_size // 2), bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Compute flatten size
        probe = torch.zeros(1, 1, n_channels, n_samples)
        probe = self.temporal_conv(probe)
        probe = self.depthwise_conv(probe)
        probe = self.separable_conv(probe)
        flat_size = probe.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, n_channels, n_samples)"""
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        return self.classifier(x)


# ─── Neural Transformer ───────────────────────────────────────────────────────

class NeuralTransformer(nn.Module):
    """
    Transformer-based neural intent decoder.
    Treats neural feature vectors as a sequence of time-step tokens.

    Input: (batch, n_timesteps, n_features)
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        n_features: int = 128,
        n_classes: int = 5,
        n_heads: int = 8,
        n_layers: int = 4,
        d_model: int = 256,
        d_ff: int = 512,
        max_seq_len: int = 200,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = self._build_positional_encoding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    @staticmethod
    def _build_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, n_timesteps, n_features)
        Returns logits (batch, n_classes) and optionally attention weights.
        """
        B, T, _ = x.shape
        x = self.input_proj(x)

        # Add positional encoding
        pe = self.pos_encoding[:, :T, :].to(x.device)
        x = x + pe

        # Prepend CLS token
        cls = self.cls_token.expand(B, 1, self.d_model)
        x = torch.cat([cls, x], dim=1)

        # Transformer encoding
        x = self.transformer(x)
        x = self.norm(x[:, 0])  # CLS token output

        logits = self.head(x)
        return logits


# ─── MC Dropout for Bayesian Uncertainty ─────────────────────────────────────

class MCDropoutPredictor:
    """
    Monte Carlo Dropout for Bayesian uncertainty estimation.
    Runs N forward passes with dropout enabled to estimate uncertainty.
    """

    def __init__(self, model: nn.Module, n_samples: int = 30) -> None:
        self.model = model
        self.n_samples = n_samples

    def _enable_dropout(self) -> None:
        """Enable dropout layers at inference time."""
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> tuple[NDArray, float, float]:
        """
        Returns (mean_probs, epistemic_uncertainty, aleatoric_uncertainty).
        """
        self.model.eval()
        self._enable_dropout()

        all_probs = []
        for _ in range(self.n_samples):
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        all_probs_arr = np.array(all_probs)  # (n_samples, batch, n_classes)

        # Mean prediction
        mean_probs = all_probs_arr.mean(axis=0).squeeze()

        # Epistemic uncertainty: variance across MC samples (predictive variance)
        epistemic = float(all_probs_arr.var(axis=0).mean())

        # Aleatoric: mean of individual entropies
        entropies = -np.sum(all_probs_arr * np.log(all_probs_arr + 1e-12), axis=-1)
        aleatoric = float(entropies.mean())

        return mean_probs, epistemic, aleatoric


# ─── Predix OmniMind Oracle ───────────────────────────────────────────────────

class PredixOmniMind:
    """
    NeuroForge Nexus — Predix OmniMind: The Sentient Neural AI Oracle.

    Ensemble of EEGNet + NeuralTransformer with Monte Carlo Dropout uncertainty.
    Provides real-time intent prediction, cognitive state assessment, and
    adaptive neurofeedback recommendations.

    Usage:
        oracle = PredixOmniMind(paradigm=BCIParadigm.MOTOR_IMAGERY)
        intent = await oracle.predict_intent(features)
        profile = await oracle.assess_cognitive_state(eeg_epoch)
    """

    MOTOR_LABELS = ["rest", "left_hand", "right_hand", "feet", "tongue"]
    COGNITIVE_THRESHOLDS = {
        "attention": (0.7, 0.9),
        "flow": (0.8, 1.0),
        "stress": (0.0, 0.4),
    }

    def __init__(
        self,
        paradigm: BCIParadigm = BCIParadigm.MOTOR_IMAGERY,
        n_channels: int = 64,
        n_samples: int = 256,
        sample_rate: int = 256,
        model_size: str = "base",
        device: Optional[str] = None,
    ) -> None:
        self.paradigm = paradigm
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.model_size = model_size

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        n_classes = len(self.MOTOR_LABELS)

        # EEGNet branch
        self.eegnet = EEGNet(
            n_classes=n_classes,
            n_channels=n_channels,
            n_samples=n_samples,
            sample_rate=sample_rate,
        ).to(self.device)

        # Transformer branch
        n_features = 128 if model_size == "base" else 256
        self.transformer = NeuralTransformer(
            n_features=n_features,
            n_classes=n_classes,
            n_heads=8 if model_size == "large" else 4,
            n_layers=4 if model_size == "large" else 2,
        ).to(self.device)

        # MC Dropout wrappers
        self.eegnet_mc = MCDropoutPredictor(self.eegnet)
        self.transformer_mc = MCDropoutPredictor(self.transformer)

        # Cognitive state heads (lightweight linear probes)
        self.attention_head = nn.Linear(n_features, 1).to(self.device)
        self.workload_head = nn.Linear(n_features, 1).to(self.device)
        self.emotion_head = nn.Linear(n_features, 2).to(self.device)  # valence, arousal

        # Ensemble weights (learnable)
        self.ensemble_logit_weights = nn.Parameter(torch.ones(2) / 2.0)

        self._inference_count = 0
        self._total_latency_ms = 0.0

        logger.info(
            f"PredixOmniMind initialized: {paradigm.value} | "
            f"device={self.device} | size={model_size} | "
            f"channels={n_channels} | classes={n_classes}"
        )

    @torch.no_grad()
    async def predict_intent(
        self,
        features: NDArray[np.float64],
        raw_epoch: Optional[NDArray[np.float64]] = None,
    ) -> NeuralPrediction:
        """
        Async intent prediction from neural features.

        features: (n_features,) or (n_timesteps, n_features)
        raw_epoch: optional (n_channels, n_samples) for EEGNet branch

        Returns NeuralPrediction with uncertainty estimates.
        """
        start = time.perf_counter()

        # ── EEGNet branch ────────────────────────────────────────────────────
        if raw_epoch is not None:
            eeg_tensor = torch.tensor(raw_epoch, dtype=torch.float32)
            eeg_tensor = eeg_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, n_ch, n_samp)
            eeg_tensor = eeg_tensor.to(self.device)
            eeg_probs, eeg_epi, eeg_ale = self.eegnet_mc.predict_with_uncertainty(eeg_tensor)
        else:
            eeg_probs = None
            eeg_epi, eeg_ale = 0.0, 0.0

        # ── Transformer branch ───────────────────────────────────────────────
        if features.ndim == 1:
            feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Pad/truncate features to expected transformer input size
        expected_feat = self.transformer.input_proj.in_features
        if feat_tensor.shape[-1] != expected_feat:
            feat_tensor = F.pad(feat_tensor, (0, max(0, expected_feat - feat_tensor.shape[-1])))
            feat_tensor = feat_tensor[..., :expected_feat]
        feat_tensor = feat_tensor.to(self.device)

        trans_probs, trans_epi, trans_ale = self.transformer_mc.predict_with_uncertainty(feat_tensor)

        # ── Ensemble ─────────────────────────────────────────────────────────
        weights = F.softmax(self.ensemble_logit_weights, dim=0).detach().cpu().numpy()
        if eeg_probs is not None:
            final_probs = (
                weights[0] * np.atleast_1d(eeg_probs) +
                weights[1] * np.atleast_1d(trans_probs)
            )
            uncertainty = float(weights[0] * eeg_epi + weights[1] * trans_epi)
            aleatoric = float(weights[0] * eeg_ale + weights[1] * trans_ale)
        else:
            final_probs = np.atleast_1d(trans_probs)
            uncertainty = trans_epi
            aleatoric = trans_ale

        # Smooth probabilities to sum to 1
        final_probs = np.abs(final_probs)
        if final_probs.sum() > 0:
            final_probs /= final_probs.sum()
        else:
            final_probs = np.ones(len(self.MOTOR_LABELS)) / len(self.MOTOR_LABELS)

        best = int(np.argmax(final_probs))
        labels = self.MOTOR_LABELS

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._inference_count += 1
        self._total_latency_ms += elapsed_ms

        return NeuralPrediction(
            label=labels[best],
            class_id=best,
            confidence=float(final_probs[best]),
            uncertainty=uncertainty,
            aleatoric=aleatoric,
            probabilities={l: float(p) for l, p in zip(labels, final_probs)},
            latency_ms=elapsed_ms,
            timestamp=time.time(),
        )

    @torch.no_grad()
    async def assess_cognitive_state(
        self, features: NDArray[np.float64]
    ) -> CognitiveProfile:
        """
        Assess real-time cognitive state from neural features.
        Returns a CognitiveProfile with attention, workload, emotion, flow.
        """
        feat_tensor = torch.tensor(
            np.atleast_1d(features)[:128], dtype=torch.float32
        ).to(self.device)

        # Pad to expected input size
        if feat_tensor.shape[0] < 128:
            feat_tensor = F.pad(feat_tensor, (0, 128 - feat_tensor.shape[0]))

        # Cognitive index predictions
        attention = float(torch.sigmoid(self.attention_head(feat_tensor)).item())
        workload = float(torch.sigmoid(self.workload_head(feat_tensor)).item())

        emotion_out = torch.tanh(self.emotion_head(feat_tensor))
        valence = float(emotion_out[0].item())
        arousal = float(torch.sigmoid(emotion_out[1]).item())

        # Derived states
        flow_prob = float(np.clip(attention * (1 - workload * 0.3) * (1 - abs(valence) * 0.2), 0, 1))
        stress = float(np.clip(workload * 0.6 + arousal * 0.4 - attention * 0.3, 0, 1))
        creativity = float(np.clip(attention * 0.4 + flow_prob * 0.4 + valence * 0.2, 0, 1))
        memory_load = float(np.clip(workload * 0.7 + attention * 0.3, 0, 1))

        # Determine dominant state
        state = self._classify_state(attention, workload, stress, flow_prob)

        return CognitiveProfile(
            state=state,
            attention_index=attention,
            workload_index=workload,
            stress_level=stress,
            flow_probability=flow_prob,
            creativity_index=creativity,
            memory_load=memory_load,
            emotional_valence=valence,
            arousal=arousal,
            iq_amplification=float(np.clip(flow_prob * 0.8 + attention * 0.2, 0, 1)),
        )

    def _classify_state(
        self,
        attention: float,
        workload: float,
        stress: float,
        flow: float,
    ) -> CognitiveState:
        """Rule-based cognitive state classification."""
        if flow > 0.8 and stress < 0.2:
            return CognitiveState.FLOW
        if attention > 0.85 and workload > 0.7:
            return CognitiveState.HYPERFOCUSED
        if attention > 0.7:
            return CognitiveState.FOCUSED
        if stress > 0.6:
            return CognitiveState.STRESSED
        if attention < 0.3 and workload < 0.3:
            return CognitiveState.RELAXED
        return CognitiveState.FOCUSED

    def mean_inference_latency(self) -> float:
        """Returns mean inference latency in milliseconds."""
        if self._inference_count == 0:
            return 0.0
        return self._total_latency_ms / self._inference_count

    def get_model_stats(self) -> dict:
        """Return model size and parameter counts."""
        eegnet_params = sum(p.numel() for p in self.eegnet.parameters())
        trans_params = sum(p.numel() for p in self.transformer.parameters())
        return {
            "eegnet_params": eegnet_params,
            "transformer_params": trans_params,
            "total_params": eegnet_params + trans_params,
            "device": str(self.device),
            "model_size": self.model_size,
            "inference_count": self._inference_count,
            "mean_latency_ms": self.mean_inference_latency(),
        }


# ─── Cognitive Amplifier ─────────────────────────────────────────────────────

class CognitiveAmplifier:
    """
    NeuroForge Nexus Cognitive Amplification Engine.

    Real-time neurofeedback system that:
    1. Assesses current cognitive state via OmniMind
    2. Computes optimal neurostimulation parameters
    3. Issues feedback commands to actuator layer
    4. Tracks session progress and adaptation
    """

    STIMULATION_PROTOCOLS = {
        CognitiveState.FOCUSED: {
            "alpha_suppression": True,
            "beta_enhancement": True,
            "target_frequency_hz": 18.0,
            "pulse_width_us": 200,
            "amplitude_ma": 0.5,
        },
        CognitiveState.FLOW: {
            "theta_enhancement": True,
            "gamma_synchrony": True,
            "target_frequency_hz": 40.0,
            "pulse_width_us": 100,
            "amplitude_ma": 0.3,
        },
        CognitiveState.STRESSED: {
            "alpha_enhancement": True,
            "beta_suppression": True,
            "target_frequency_hz": 10.0,
            "pulse_width_us": 500,
            "amplitude_ma": 0.8,
        },
        CognitiveState.CREATIVE: {
            "theta_enhancement": True,
            "alpha_peak_shift": True,
            "target_frequency_hz": 7.5,
            "pulse_width_us": 300,
            "amplitude_ma": 0.4,
        },
    }

    def __init__(self, oracle: PredixOmniMind) -> None:
        self.oracle = oracle
        self.session_history: list[CognitiveProfile] = []
        self._session_start: Optional[float] = None

    def start_session(self) -> str:
        """Begin a new amplification session. Returns session ID."""
        import uuid
        session_id = str(uuid.uuid4())
        self._session_start = time.time()
        self.session_history.clear()
        logger.info(f"CognitiveAmplifier session {session_id} started")
        return session_id

    async def amplify_step(
        self, features: NDArray[np.float64], target_state: CognitiveState = CognitiveState.FLOW
    ) -> dict:
        """
        One amplification step: assess → compare → recommend.
        Returns neurofeedback command dict.
        """
        profile = await self.oracle.assess_cognitive_state(features)
        self.session_history.append(profile)

        # Determine gap between current and target
        current_flow = profile.flow_probability
        current_attention = profile.attention_index
        current_stress = profile.stress_level

        # Build neurofeedback command
        protocol = self.STIMULATION_PROTOCOLS.get(target_state, {})

        # Adaptive amplitude scaling based on gap
        flow_gap = max(0, 0.8 - current_flow)
        attention_gap = max(0, 0.7 - current_attention)
        amplitude_scale = min(1.0 + flow_gap + attention_gap, 2.0)

        command = {
            "timestamp": time.time(),
            "current_state": profile.state.name,
            "target_state": target_state.name,
            "protocol": protocol,
            "amplitude_scale": amplitude_scale,
            "feedback_type": self._select_feedback_type(profile, target_state),
            "duration_ms": 500,
            "metrics": {
                "attention": profile.attention_index,
                "flow": profile.flow_probability,
                "stress": profile.stress_level,
                "iq_boost": profile.iq_amplification,
            },
        }
        return command

    def _select_feedback_type(
        self, profile: CognitiveProfile, target: CognitiveState
    ) -> str:
        """Select neurofeedback modality based on state delta."""
        if profile.stress_level > 0.6:
            return "binaural_beats_alpha"
        if profile.flow_probability < 0.3:
            return "tacs_gamma_40hz"
        if profile.attention_index < 0.5:
            return "tdcs_prefrontal_anode"
        return "neurofeedback_visual"

    def get_session_summary(self) -> dict:
        """Summarize current session metrics."""
        if not self.session_history:
            return {}
        attentions = [p.attention_index for p in self.session_history]
        flows = [p.flow_probability for p in self.session_history]
        stresses = [p.stress_level for p in self.session_history]
        duration = time.time() - (self._session_start or time.time())

        return {
            "duration_s": duration,
            "n_steps": len(self.session_history),
            "mean_attention": float(np.mean(attentions)),
            "peak_attention": float(np.max(attentions)),
            "mean_flow": float(np.mean(flows)),
            "peak_flow": float(np.max(flows)),
            "mean_stress": float(np.mean(stresses)),
            "performance_gain_pct": float((np.mean(flows[-10:]) - np.mean(flows[:10])) * 100)
            if len(flows) >= 20 else 0.0,
        }
