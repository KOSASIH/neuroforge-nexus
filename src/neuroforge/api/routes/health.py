"""Health endpoints."""
import time
from fastapi import APIRouter
from neuroforge.api.models import HealthResponse
from neuroforge.__version__ import __version__

router = APIRouter()
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health():
    from neuroforge.api.main import app_state
    return HealthResponse(
        status="healthy",
        version=__version__,
        uptime_s=round(time.time() - _start_time, 1),
        node_id=app_state.network.node_id,
        active_ws_streams=len(app_state.active_ws_connections),
        active_peers=len(app_state.network.get_active_peers()),
        mean_inference_latency_ms=round(app_state.omnimind.mean_inference_latency(), 3),
        encryption_stats=app_state.encryption_engine.get_stats(),
        model_stats=app_state.omnimind.get_model_stats(),
        quantum_qubits=app_state.quantum_optimizer.n_qubits,
        timestamp=time.time(),
    )


@router.get("/health/ping")
async def ping():
    return {"pong": time.time(), "status": "online"}
