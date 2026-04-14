"""Neural signal processing endpoints."""
import time
import numpy as np
from fastapi import APIRouter, HTTPException
from neuroforge.api.models import DecodeRequest, DecodeResponse

router = APIRouter()


@router.post("/decode", response_model=DecodeResponse)
async def decode_neural(req: DecodeRequest):
    from neuroforge.api.main import app_state
    start = time.perf_counter()

    try:
        data = np.array(req.data, dtype=np.float64)
        if data.ndim != 2:
            raise HTTPException(400, "data must be 2D: (n_channels, n_samples)")

        sp = app_state.signal_processor
        sp_instance = app_state.signal_processor

        if req.preprocess:
            processed, artifacts = sp_instance.full_preprocessing(data)
        else:
            processed, artifacts = data, []

        snr = sp.compute_snr(processed)
        qs = sp.quality_score(processed)

        # Feature extraction
        features = sp.extract_features(processed)
        feat_vec = np.concatenate([features.band_features, features.hilbert_amplitude])

        # Decode intent
        intent = app_state.neural_decoder.decode(feat_vec)

        latency = (time.perf_counter() - start) * 1000

        return DecodeResponse(
            label=intent.label,
            class_id=intent.class_id,
            confidence=round(intent.confidence, 4),
            probabilities={k: round(v, 4) for k, v in intent.probabilities.items()},
            quality_score=round(qs, 3),
            artifacts=artifacts,
            snr_db=round(snr, 2),
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/preprocess")
async def preprocess(req: DecodeRequest):
    """Apply full preprocessing pipeline and return filtered signal."""
    from neuroforge.api.main import app_state
    try:
        data = np.array(req.data, dtype=np.float64)
        sp = app_state.signal_processor
        processed, artifacts = sp.full_preprocessing(data)
        return {
            "shape": list(processed.shape),
            "artifacts": artifacts,
            "snr_db": round(sp.compute_snr(processed), 2),
            "quality": round(sp.quality_score(processed), 3),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/features")
async def extract_features(req: DecodeRequest):
    """Extract full feature set from neural epoch."""
    from neuroforge.api.main import app_state
    try:
        data = np.array(req.data, dtype=np.float64)
        sp = app_state.signal_processor
        if req.preprocess:
            data, _ = sp.full_preprocessing(data)
        features = sp.extract_spectral_features(data)
        return {
            "band_power": {k: v.tolist() for k, v in features.band_power.items()},
            "spectral_entropy": features.spectral_entropy.tolist(),
            "dominant_frequency": features.dominant_frequency.tolist(),
        }
    except Exception as e:
        raise HTTPException(500, str(e))
