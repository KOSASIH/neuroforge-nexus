"""Quantum optimization endpoints."""
import numpy as np
from fastapi import APIRouter, HTTPException
from neuroforge.api.models import QuantumOptimizeRequest, QuantumOptimizeResponse

router = APIRouter()


@router.post("/optimize", response_model=QuantumOptimizeResponse)
async def quantum_optimize(req: QuantumOptimizeRequest):
    from neuroforge.api.main import app_state, QUANTUM_OPS
    try:
        matrix = np.array(req.connectivity_matrix, dtype=np.float64)
        QUANTUM_OPS.labels(method=req.method).inc()
        result = app_state.quantum_optimizer.optimize_neural_weights(matrix, req.method)
        return QuantumOptimizeResponse(
            optimal_params=result.optimal_params.tolist(),
            optimal_value=round(result.optimal_value, 6),
            iterations=result.iterations,
            convergence=result.convergence_history,
            quantum_advantage_estimate=round(result.quantum_advantage_estimate, 2),
            runtime_ms=round(result.runtime_ms, 2),
            method=result.method,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/status")
async def quantum_status():
    from neuroforge.api.main import app_state
    return {
        "n_qubits": app_state.quantum_optimizer.n_qubits,
        "circuit_depth": app_state.quantum_optimizer.depth,
        "noise_model": "depolarizing",
        "backend": "statevector_simulator",
        "encryption": app_state.encryption_engine.get_stats(),
    }


@router.post("/random-bits")
async def quantum_random_bits(n_bits: int = 256):
    from neuroforge.api.main import app_state
    bits = app_state.quantum_optimizer.quantum_random_sample(n_bits)
    return {"n_bits": n_bits, "bits": bits.tolist(), "hex": bits.tobytes().hex()}
