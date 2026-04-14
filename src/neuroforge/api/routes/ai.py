"""AI / OmniMind endpoints."""
import numpy as np
from fastapi import APIRouter, HTTPException
from neuroforge.api.models import PredictRequest, PredictResponse, AmplifyRequest, AmplifyResponse
from neuroforge.core.constants import CognitiveState

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict_intent(req: PredictRequest):
    from neuroforge.api.main import app_state
    try:
        features = np.array(req.features, dtype=np.float64)
        raw_epoch = (
            np.array(req.raw_epoch, dtype=np.float64) if req.raw_epoch else None
        )
        pred = await app_state.omnimind.predict_intent(features, raw_epoch)
        return PredictResponse(
            label=pred.label,
            class_id=pred.class_id,
            confidence=round(pred.confidence, 4),
            uncertainty=round(pred.uncertainty, 6),
            aleatoric=round(pred.aleatoric, 6),
            probabilities={k: round(v, 4) for k, v in pred.probabilities.items()},
            latency_ms=round(pred.latency_ms, 2),
            model_version=pred.model_version,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/cognitive-state")
async def assess_cognitive_state(req: PredictRequest):
    from neuroforge.api.main import app_state
    try:
        features = np.array(req.features, dtype=np.float64)
        profile = await app_state.omnimind.assess_cognitive_state(features)
        return {
            "state": profile.state.name,
            "attention_index": round(profile.attention_index, 3),
            "workload_index": round(profile.workload_index, 3),
            "stress_level": round(profile.stress_level, 3),
            "flow_probability": round(profile.flow_probability, 3),
            "creativity_index": round(profile.creativity_index, 3),
            "memory_load": round(profile.memory_load, 3),
            "emotional_valence": round(profile.emotional_valence, 3),
            "arousal": round(profile.arousal, 3),
            "iq_amplification": round(profile.iq_amplification, 3),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/amplify", response_model=AmplifyResponse)
async def amplify(req: AmplifyRequest):
    from neuroforge.api.main import app_state
    try:
        features = np.array(req.features, dtype=np.float64)
        try:
            target = CognitiveState[req.target_state.upper()]
        except KeyError:
            target = CognitiveState.FLOW

        cmd = await app_state.cognitive_amplifier.amplify_step(features, target)
        summary = app_state.cognitive_amplifier.get_session_summary()

        return AmplifyResponse(
            feedback_type=cmd.get("feedback_type", ""),
            current_state=cmd.get("current_state", ""),
            target_state=cmd.get("target_state", ""),
            amplitude_scale=round(float(cmd.get("amplitude_scale", 1.0)), 3),
            metrics={k: round(v, 3) for k, v in cmd.get("metrics", {}).items()},
            session_summary=summary,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/models/stats")
async def model_stats():
    from neuroforge.api.main import app_state
    return app_state.omnimind.get_model_stats()
