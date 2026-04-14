"""
NeuroForge Nexus — API Pydantic Models
=======================================
Request/response schemas for all REST endpoints.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


# ─── Neural ───────────────────────────────────────────────────────────────────

class DecodeRequest(BaseModel):
    data: list[list[float]] = Field(..., description="(n_channels, n_samples) float data")
    sample_rate: int = Field(256, description="Sample rate in Hz")
    paradigm: str = Field("motor_imagery", description="BCI paradigm")
    preprocess: bool = Field(True, description="Apply full preprocessing pipeline")


class DecodeResponse(BaseModel):
    label: str
    class_id: int
    confidence: float
    probabilities: dict[str, float]
    quality_score: float
    artifacts: list[str]
    snr_db: float
    latency_ms: float


# ─── AI ───────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="Flattened feature vector")
    raw_epoch: Optional[list[list[float]]] = Field(None, description="Optional raw EEG (n_ch, n_samp)")
    paradigm: str = Field("motor_imagery")


class PredictResponse(BaseModel):
    label: str
    class_id: int
    confidence: float
    uncertainty: float
    aleatoric: float
    probabilities: dict[str, float]
    latency_ms: float
    model_version: str


class AmplifyRequest(BaseModel):
    features: list[float] = Field(..., description="Cognitive feature vector")
    target_state: str = Field("FLOW", description="Target cognitive state name")
    duration_steps: int = Field(1, ge=1, le=100)


class AmplifyResponse(BaseModel):
    feedback_type: str
    current_state: str
    target_state: str
    amplitude_scale: float
    metrics: dict[str, float]
    session_summary: dict[str, Any]


# ─── Quantum ──────────────────────────────────────────────────────────────────

class QuantumOptimizeRequest(BaseModel):
    connectivity_matrix: list[list[float]] = Field(
        ..., description="Neural connectivity weight matrix"
    )
    method: str = Field("qaoa", description="'qaoa' or 'annealing'")
    n_iterations: int = Field(50, ge=10, le=500)


class QuantumOptimizeResponse(BaseModel):
    optimal_params: list[float]
    optimal_value: float
    iterations: int
    convergence: list[float]
    quantum_advantage_estimate: float
    runtime_ms: float
    method: str


# ─── Simulation ───────────────────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    paradigm: str = Field("motor_imagery", description="Paradigm to simulate")
    n_trials: int = Field(20, ge=1, le=500)
    duration_s: float = Field(60.0, ge=1.0, le=3600.0)
    n_channels: int = Field(64, ge=1, le=256)
    sample_rate: int = Field(256, ge=64, le=2000)
    snr_db: float = Field(15.0, ge=0.0, le=60.0)
    cognitive_state: str = Field("relaxed")


class SimulateResponse(BaseModel):
    session_id: str
    subject_id: str
    paradigm: str
    n_channels: int
    n_samples: int
    duration_s: float
    n_events: int
    snr_db: float
    sample_rate: int
    data_shape: list[int]


# ─── Network ──────────────────────────────────────────────────────────────────

class BroadcastRequest(BaseModel):
    payload: dict[str, Any]
    state_type: str = Field("cognitive_profile")


class BroadcastResponse(BaseModel):
    message_id: str
    peers_notified: int
    timestamp: float


class NodeRegistrationRequest(BaseModel):
    node_id: str
    address: str
    port: int = 7777
    capabilities: list[str] = Field(default_factory=lambda: ["eeg"])
    n_channels: int = 64
    sample_rate: int = 1000
    firmware_version: str = "nexus-fw-1.0.0"


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_s: float
    node_id: str
    active_ws_streams: int
    active_peers: int
    mean_inference_latency_ms: float
    encryption_stats: dict[str, Any]
    model_stats: dict[str, Any]
    quantum_qubits: int
    timestamp: float
