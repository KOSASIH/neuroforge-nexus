"""
NeuroForge Nexus — FastAPI Application
=======================================
Production-grade REST + WebSocket API for the NeuroForge BCI platform.

Endpoints:
  /health           — System health & neural latency stats
  /neural/*         — Signal processing & neural decoding
  /ai/*             — AI intent prediction & cognitive amplification
  /quantum/*        — Quantum optimization & encryption
  /network/*        — Neural mesh network management
  /sim/*            — Brain simulation & synthetic data
  /ws/neural-stream — Real-time WebSocket neural data stream
  /ws/omnimind      — Live AI cognitive assistance WebSocket
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from neuroforge.__version__ import __version__
from neuroforge.core.signal_processor import SignalProcessor
from neuroforge.core.neural_encoder import NeuralEncoder, NeuralDecoder
from neuroforge.core.constants import BCIParadigm, CognitiveState
from neuroforge.ai.predix_omnimind import PredixOmniMind, CognitiveAmplifier
from neuroforge.quantum.quantum_optimizer import QuantumOptimizer
from neuroforge.quantum.quantum_encryption import NeuralEncryptionEngine
from neuroforge.network.teleforge_network import TeleForgeNetwork, NexusNode, NodeStatus
from neuroforge.simulation.brain_simulator import BrainSimulator
from neuroforge.api.models import (
    DecodeRequest, DecodeResponse,
    PredictRequest, PredictResponse,
    AmplifyRequest, AmplifyResponse,
    QuantumOptimizeRequest, QuantumOptimizeResponse,
    SimulateRequest, SimulateResponse,
    BroadcastRequest, BroadcastResponse,
    NodeRegistrationRequest,
    HealthResponse,
)


# ─── Prometheus Metrics ───────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "neuroforge_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "neuroforge_request_latency_seconds",
    "API request latency",
    ["endpoint"],
)
NEURAL_INFERENCES = Counter(
    "neuroforge_neural_inferences_total",
    "Total neural inference calls",
)
ACTIVE_STREAMS = Gauge(
    "neuroforge_active_streams",
    "Currently active WebSocket neural streams",
)
QUANTUM_OPS = Counter(
    "neuroforge_quantum_ops_total",
    "Total quantum optimization operations",
    ["method"],
)


# ─── Application State ────────────────────────────────────────────────────────

class AppState:
    """Global application state shared across requests."""
    signal_processor: SignalProcessor
    neural_encoder: NeuralEncoder
    neural_decoder: NeuralDecoder
    omnimind: PredixOmniMind
    cognitive_amplifier: CognitiveAmplifier
    quantum_optimizer: QuantumOptimizer
    encryption_engine: NeuralEncryptionEngine
    network: TeleForgeNetwork
    brain_simulator: BrainSimulator
    active_ws_connections: set[str]


app_state = AppState()


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and teardown NeuroForge platform components."""
    logger.info("🧠 NeuroForge Nexus API starting up...")

    # Initialize core components
    app_state.signal_processor = SignalProcessor(sample_rate=1000, n_channels=64)
    app_state.neural_encoder = NeuralEncoder(sample_rate=30000)
    app_state.neural_decoder = NeuralDecoder(BCIParadigm.MOTOR_IMAGERY)
    app_state.omnimind = PredixOmniMind(n_channels=64, n_samples=256, sample_rate=256)
    app_state.cognitive_amplifier = CognitiveAmplifier(app_state.omnimind)
    app_state.quantum_optimizer = QuantumOptimizer(n_qubits=8, depth=3)
    app_state.encryption_engine = NeuralEncryptionEngine()
    app_state.network = TeleForgeNetwork(
        node_id=f"nexus-api-{str(uuid.uuid4())[:8]}"
    )
    app_state.brain_simulator = BrainSimulator(n_channels=64, sample_rate=256)
    app_state.active_ws_connections = set()

    # Start network
    await app_state.network.start()

    logger.info(f"🚀 NeuroForge Nexus v{__version__} ONLINE")

    yield  # ─── App running ───

    logger.info("⏹  NeuroForge Nexus API shutting down...")
    await app_state.network.stop()


# ─── App Factory ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="NeuroForge Nexus API",
    description=(
        "🧠 NeuroForge Nexus — Pinnacle BCI Platform API. "
        "Real-time neural signal processing, AI cognitive augmentation, "
        "quantum optimization, and planetary neural mesh networking."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Import and mount routers
from neuroforge.api.routes import neural, ai, quantum, network, health as health_router
app.include_router(health_router.router, prefix="", tags=["Health"])
app.include_router(neural.router, prefix="/neural", tags=["Neural Signal Processing"])
app.include_router(ai.router, prefix="/ai", tags=["AI / OmniMind"])
app.include_router(quantum.router, prefix="/quantum", tags=["Quantum"])
app.include_router(network.router, prefix="/network", tags=["Neural Mesh Network"])


# ─── Prometheus Metrics Endpoint ──────────────────────────────────────────────

@app.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ─── WebSocket: Neural Data Stream ────────────────────────────────────────────

@app.websocket("/ws/neural-stream")
async def neural_stream_ws(websocket: WebSocket):
    """
    Real-time neural data stream WebSocket.

    Client sends: {"command": "start"|"stop"|"calibrate", "config": {...}}
    Server sends: JSON frames with processed neural features
    """
    await websocket.accept()
    conn_id = str(uuid.uuid4())[:8]
    app_state.active_ws_connections.add(conn_id)
    ACTIVE_STREAMS.inc()

    logger.info(f"WS neural-stream connected: {conn_id}")

    try:
        sp = app_state.signal_processor
        streaming = False
        frame_count = 0

        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.05)
                command = msg.get("command", "")

                if command == "start":
                    streaming = True
                    await websocket.send_json({"status": "streaming", "connection_id": conn_id})

                elif command == "stop":
                    streaming = False
                    await websocket.send_json({"status": "stopped", "frames": frame_count})

                elif command == "calibrate":
                    await websocket.send_json({
                        "status": "calibrating",
                        "message": "CSP calibration requires labeled training data",
                    })

                elif command == "ping":
                    await websocket.send_json({"pong": time.time()})

            except asyncio.TimeoutError:
                pass  # No incoming message, continue streaming

            if streaming:
                # Generate synthetic frame for demo (replace with hardware ADC read)
                n_ch, n_samp = 64, 256
                raw = (np.random.randn(n_ch, n_samp) * 10e-6).astype(np.float32)
                processed, artifacts = sp.full_preprocessing(raw.astype(np.float64))
                snr = sp.compute_snr(processed)
                qs = sp.quality_score(processed)

                # Compute band powers for top 4 channels
                freqs, psd = sp.compute_psd(processed[:4])
                band_pwr = sp.band_power(freqs, psd)

                frame = {
                    "frame": frame_count,
                    "timestamp": time.time(),
                    "channels": n_ch,
                    "samples": n_samp,
                    "snr_db": round(snr, 2),
                    "quality": round(qs, 3),
                    "artifacts": artifacts,
                    "band_power": {
                        k: v[:4].tolist() for k, v in band_pwr.items()
                    },
                }
                await websocket.send_json(frame)
                frame_count += 1
                await asyncio.sleep(0.25)  # 4 Hz frame rate

    except WebSocketDisconnect:
        logger.info(f"WS neural-stream disconnected: {conn_id}")
    finally:
        app_state.active_ws_connections.discard(conn_id)
        ACTIVE_STREAMS.dec()


# ─── WebSocket: OmniMind Live Assistance ──────────────────────────────────────

@app.websocket("/ws/omnimind")
async def omnimind_ws(websocket: WebSocket):
    """
    Live OmniMind AI cognitive assistance WebSocket.

    Client sends: {"type": "features", "data": [...], "shape": [T, F]}
    Server sends: {"intent": ..., "cognitive_state": ..., "amplification": ...}
    """
    await websocket.accept()
    conn_id = str(uuid.uuid4())[:8]
    logger.info(f"WS OmniMind connected: {conn_id}")

    oracle = app_state.omnimind
    amplifier = app_state.cognitive_amplifier
    session_id = amplifier.start_session()

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type", "")

            if msg_type == "features":
                # Decode incoming neural features
                shape = msg.get("shape", [1, 128])
                data = np.array(msg.get("data", []), dtype=np.float64).reshape(shape)

                features = data[0] if data.ndim == 2 else data
                prediction = await oracle.predict_intent(features)
                profile = await oracle.assess_cognitive_state(features)
                amp_cmd = await amplifier.amplify_step(features, CognitiveState.FLOW)

                NEURAL_INFERENCES.inc()

                await websocket.send_json({
                    "type": "omnimind_response",
                    "timestamp": time.time(),
                    "intent": {
                        "label": prediction.label,
                        "confidence": round(prediction.confidence, 4),
                        "uncertainty": round(prediction.uncertainty, 4),
                        "probabilities": {k: round(v, 4) for k, v in prediction.probabilities.items()},
                        "latency_ms": round(prediction.latency_ms, 2),
                    },
                    "cognitive_state": {
                        "state": profile.state.name,
                        "attention": round(profile.attention_index, 3),
                        "flow": round(profile.flow_probability, 3),
                        "stress": round(profile.stress_level, 3),
                        "iq_amplification": round(profile.iq_amplification, 3),
                    },
                    "amplification": {
                        "feedback_type": amp_cmd.get("feedback_type"),
                        "target_state": amp_cmd.get("target_state"),
                        "amplitude_scale": round(amp_cmd.get("amplitude_scale", 1.0), 3),
                    },
                    "session_id": session_id,
                })

            elif msg_type == "session_summary":
                summary = amplifier.get_session_summary()
                await websocket.send_json({
                    "type": "session_summary",
                    "session_id": session_id,
                    "summary": summary,
                })

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})

    except WebSocketDisconnect:
        logger.info(f"WS OmniMind disconnected: {conn_id}")


# ─── Root ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "NeuroForge Nexus API",
        "version": __version__,
        "status": "online",
        "tagline": "Where Minds Become Mythic Forges 🧠",
        "docs": "/docs",
        "metrics": "/metrics",
        "websockets": ["/ws/neural-stream", "/ws/omnimind"],
    }
