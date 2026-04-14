"""Neural mesh network endpoints."""
from fastapi import APIRouter, HTTPException
from neuroforge.api.models import BroadcastRequest, BroadcastResponse, NodeRegistrationRequest
from neuroforge.network.teleforge_network import NexusNode, NodeStatus
import time

router = APIRouter()


@router.get("/nodes")
async def list_nodes():
    from neuroforge.api.main import app_state
    peers = app_state.network.get_active_peers()
    return {
        "count": len(peers),
        "nodes": [
            {
                "node_id": p.node_id,
                "address": p.address,
                "status": p.status.value,
                "latency_ms": round(p.latency_ms, 1),
                "capabilities": p.capabilities,
                "last_seen": p.last_seen,
            }
            for p in peers
        ],
    }


@router.post("/nodes/register")
async def register_node(req: NodeRegistrationRequest):
    from neuroforge.api.main import app_state
    node = NexusNode(
        node_id=req.node_id,
        address=req.address,
        port=req.port,
        public_key=b"\x00" * 32,
        capabilities=req.capabilities,
        firmware_version=req.firmware_version,
        n_channels=req.n_channels,
        sample_rate=req.sample_rate,
        status=NodeStatus.ACTIVE,
        last_seen=time.time(),
    )
    app_state.network.register_peer(node)
    return {"registered": req.node_id, "total_peers": len(app_state.network._peers)}


@router.post("/broadcast", response_model=BroadcastResponse)
async def broadcast(req: BroadcastRequest):
    from neuroforge.api.main import app_state
    msg_id = await app_state.network.broadcast_neural_state(
        req.payload, req.state_type
    )
    return BroadcastResponse(
        message_id=msg_id,
        peers_notified=len(app_state.network.get_active_peers()),
        timestamp=time.time(),
    )


@router.post("/sessions/create")
async def create_session(participant_ids: list[str], session_type: str = "collaboration"):
    from neuroforge.api.main import app_state
    session = await app_state.network.create_mind_mesh_session(participant_ids, session_type)
    return {
        "session_id": session.session_id,
        "participants": session.participants,
        "session_type": session.session_type,
        "started_at": session.started_at,
    }


@router.get("/stats")
async def network_stats():
    from neuroforge.api.main import app_state
    return app_state.network.get_stats()
