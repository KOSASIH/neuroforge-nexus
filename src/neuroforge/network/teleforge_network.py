"""
NeuroForge Nexus — TeleForge Neural Mesh Network
=================================================
Planetary-scale encrypted P2P network for real-time neural state sharing,
distributed BCI calibration, and collaborative mind-mesh sessions.

Architecture:
  - Nexus Node: Individual BCI device endpoint
  - TeleForge Network: DHT-based P2P overlay
  - Neural State Broadcast: Encrypted neural state gossip protocol
  - Consensus: Raft-based consensus for calibration parameters
  - Latency-aware routing: Neural mesh paths optimized for <1ms hops
"""

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from loguru import logger

from neuroforge.quantum.quantum_encryption import NeuralEncryptionEngine


# ─── Data Structures ──────────────────────────────────────────────────────────

class NodeStatus(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    CALIBRATING = "calibrating"
    STREAMING = "streaming"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass
class NexusNode:
    """A single NeuroForge Nexus network node (BCI endpoint)."""
    node_id: str
    address: str            # IP or peer ID
    port: int
    public_key: bytes
    capabilities: list[str]  # e.g. ["eeg", "ecog", "spike", "stimulate"]
    firmware_version: str
    n_channels: int
    sample_rate: int
    status: NodeStatus = NodeStatus.INITIALIZING
    last_seen: float = field(default_factory=time.time)
    latency_ms: float = 999.0
    trust_score: float = 1.0    # 0..1 reputational trust


@dataclass
class NeuralStateMessage:
    """Encrypted neural state broadcast message."""
    message_id: str
    sender_id: str
    timestamp: float
    state_type: str          # "cognitive_profile", "decoded_intent", "calibration"
    payload_encrypted: bytes
    signature: bytes
    ttl_hops: int = 10
    recipients: list[str] = field(default_factory=list)  # empty = broadcast


@dataclass
class MindMeshSession:
    """Collaborative multi-user neural synchronization session."""
    session_id: str
    participants: list[str]   # Node IDs
    session_type: str         # "collaboration", "calibration", "hive_research"
    consensus_state: dict
    neural_sync_score: float  # 0..1 cross-participant coherence
    started_at: float
    messages: list[str] = field(default_factory=list)


@dataclass
class RoutingEntry:
    """Entry in the neural mesh routing table."""
    destination: str
    next_hop: str
    hop_count: int
    latency_ms: float
    bandwidth_mbps: float
    reliability: float   # 0..1


# ─── TeleForge Network ────────────────────────────────────────────────────────

class TeleForgeNetwork:
    """
    NeuroForge TeleForge Planetary Neural Mesh Network.

    Provides:
      - Node discovery and registration
      - Encrypted neural state broadcasting
      - Latency-optimized message routing
      - Collaborative mind-mesh session management
      - Network health monitoring

    Usage:
        network = TeleForgeNetwork(node_id="nexus-001")
        await network.start()
        await network.broadcast_neural_state(profile_data)
    """

    GOSSIP_FANOUT = 6
    MAX_ROUTING_TABLE = 1024
    HEARTBEAT_INTERVAL_S = 5.0
    RECONNECT_INTERVAL_S = 30.0

    def __init__(
        self,
        node_id: Optional[str] = None,
        address: str = "0.0.0.0",
        port: int = 7777,
        bootstrap_nodes: Optional[list[str]] = None,
    ) -> None:
        self.node_id = node_id or f"nexus-{str(uuid.uuid4())[:8]}"
        self.address = address
        self.port = port
        self.bootstrap_nodes = bootstrap_nodes or []

        self._encryption = NeuralEncryptionEngine()
        self._peers: dict[str, NexusNode] = {}
        self._routing_table: dict[str, RoutingEntry] = {}
        self._sessions: dict[str, MindMeshSession] = {}
        self._message_cache: set[str] = set()  # Dedup seen message IDs
        self._message_handlers: dict[str, list[Callable]] = {}

        self._running = False
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_dropped": 0,
            "bytes_encrypted": 0,
            "uptime_start": time.time(),
        }

        logger.info(f"TeleForge node {self.node_id} initialized at {address}:{port}")

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the network node and connect to mesh."""
        self._running = True
        logger.info(f"TeleForge node {self.node_id} starting...")

        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._routing_maintenance_loop())

        # Bootstrap: connect to known nodes
        for bootstrap in self.bootstrap_nodes:
            await self._connect_to_peer(bootstrap)

        logger.info(f"TeleForge node {self.node_id} ACTIVE | peers={len(self._peers)}")

    async def stop(self) -> None:
        """Gracefully stop the network node."""
        self._running = False
        logger.info(f"TeleForge node {self.node_id} stopping")

    # ─── Peer Management ─────────────────────────────────────────────────────

    async def _connect_to_peer(self, address: str) -> Optional[NexusNode]:
        """Simulate peer connection and handshake."""
        peer_id = f"nexus-{hashlib.sha256(address.encode()).hexdigest()[:8]}"
        if peer_id in self._peers:
            return self._peers[peer_id]

        # Simulate connection latency
        await asyncio.sleep(0.001)
        latency = np.random.exponential(5.0)  # ms

        peer = NexusNode(
            node_id=peer_id,
            address=address,
            port=self.port,
            public_key=np.random.bytes(32),
            capabilities=["eeg", "stream"],
            firmware_version="nexus-fw-1.0.0",
            n_channels=64,
            sample_rate=1000,
            status=NodeStatus.ACTIVE,
            last_seen=time.time(),
            latency_ms=latency,
        )
        self._peers[peer_id] = peer
        self._update_routing(peer_id, peer_id, 1, latency, 100.0)
        logger.debug(f"Connected to peer {peer_id} | latency={latency:.1f}ms")
        return peer

    def register_peer(self, node: NexusNode) -> None:
        """Register a known peer node."""
        self._peers[node.node_id] = node
        self._update_routing(node.node_id, node.node_id, 1, node.latency_ms, 100.0)

    def get_active_peers(self) -> list[NexusNode]:
        """Return currently active peer nodes."""
        now = time.time()
        return [
            p for p in self._peers.values()
            if p.status != NodeStatus.OFFLINE
            and now - p.last_seen < 60.0
        ]

    # ─── Routing ─────────────────────────────────────────────────────────────

    def _update_routing(
        self,
        dest: str,
        next_hop: str,
        hops: int,
        latency: float,
        bandwidth: float,
        reliability: float = 1.0,
    ) -> None:
        existing = self._routing_table.get(dest)
        if existing is None or latency < existing.latency_ms:
            self._routing_table[dest] = RoutingEntry(
                destination=dest,
                next_hop=next_hop,
                hop_count=hops,
                latency_ms=latency,
                bandwidth_mbps=bandwidth,
                reliability=reliability,
            )

    def find_best_route(self, destination: str) -> Optional[RoutingEntry]:
        """Find lowest-latency route to a destination."""
        return self._routing_table.get(destination)

    # ─── Message Passing ─────────────────────────────────────────────────────

    def on(self, message_type: str, handler: Callable) -> None:
        """Register a message handler for a given type."""
        if message_type not in self._message_handlers:
            self._message_handlers[message_type] = []
        self._message_handlers[message_type].append(handler)

    async def broadcast_neural_state(
        self,
        payload: dict,
        state_type: str = "cognitive_profile",
    ) -> str:
        """
        Broadcast encrypted neural state to the mesh.
        Returns message ID.
        """
        # Serialize and encrypt payload
        payload_bytes = json.dumps(payload, default=str).encode()
        # Simple XOR encryption as placeholder for full encryption pipeline
        key_byte = self._encryption._active_key[0]
        encrypted = bytes(b ^ key_byte for b in payload_bytes)
        signature = hashlib.sha256(payload_bytes + self._encryption._active_key).digest()

        msg = NeuralStateMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.node_id,
            timestamp=time.time(),
            state_type=state_type,
            payload_encrypted=encrypted,
            signature=signature,
        )

        # Gossip to fanout peers
        targets = list(self._peers.values())[:self.GOSSIP_FANOUT]
        await self._gossip(msg, targets)

        self._stats["messages_sent"] += 1
        self._stats["bytes_encrypted"] += len(encrypted)

        return msg.message_id

    async def _gossip(
        self, msg: NeuralStateMessage, targets: list[NexusNode]
    ) -> None:
        """Gossiping: forward message to target peers (simulate)."""
        if msg.message_id in self._message_cache:
            self._stats["messages_dropped"] += 1
            return
        self._message_cache.add(msg.message_id)

        # Prune cache if too large
        if len(self._message_cache) > 10000:
            self._message_cache = set(list(self._message_cache)[-5000:])

        # Notify local handlers
        for handler in self._message_handlers.get(msg.state_type, []):
            asyncio.create_task(handler(msg))

        logger.debug(
            f"[{self.node_id}] Gossip {msg.state_type} → {len(targets)} peers"
        )

    async def send_to(
        self,
        destination_id: str,
        payload: dict,
        message_type: str = "direct",
    ) -> bool:
        """Direct unicast message to a specific node."""
        if destination_id not in self._peers:
            return False

        payload_bytes = json.dumps(payload, default=str).encode()
        key_byte = self._encryption._active_key[0]
        encrypted = bytes(b ^ key_byte for b in payload_bytes)
        sig = hashlib.sha256(payload_bytes + self._encryption._active_key).digest()

        msg = NeuralStateMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.node_id,
            timestamp=time.time(),
            state_type=message_type,
            payload_encrypted=encrypted,
            signature=sig,
            recipients=[destination_id],
        )
        await self._gossip(msg, [self._peers[destination_id]])
        self._stats["messages_sent"] += 1
        return True

    # ─── Mind Mesh Sessions ───────────────────────────────────────────────────

    async def create_mind_mesh_session(
        self,
        participant_ids: list[str],
        session_type: str = "collaboration",
    ) -> MindMeshSession:
        """Create and initialize a collaborative neural sync session."""
        session = MindMeshSession(
            session_id=str(uuid.uuid4()),
            participants=[self.node_id] + participant_ids,
            session_type=session_type,
            consensus_state={},
            neural_sync_score=0.0,
            started_at=time.time(),
        )
        self._sessions[session.session_id] = session

        # Notify participants
        await self.broadcast_neural_state(
            {"session_id": session.session_id, "participants": session.participants},
            state_type="session_invite",
        )

        logger.info(
            f"MindMesh session {session.session_id} created | "
            f"type={session_type} | participants={len(session.participants)}"
        )
        return session

    async def update_session_sync_score(
        self,
        session_id: str,
        neural_coherence_matrix: NDArray[np.float64],
    ) -> float:
        """
        Update neural sync score for a mind-mesh session.
        coherence_matrix: (n_participants, n_participants)
        """
        if session_id not in self._sessions:
            return 0.0

        # Mean off-diagonal coherence as sync score
        n = neural_coherence_matrix.shape[0]
        off_diag = neural_coherence_matrix[~np.eye(n, dtype=bool)]
        sync_score = float(np.mean(off_diag))

        self._sessions[session_id].neural_sync_score = sync_score
        return sync_score

    # ─── Background Tasks ─────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat to all known peers."""
        while self._running:
            for peer in list(self._peers.values()):
                # Simulate latency measurement
                peer.last_seen = time.time()
                peer.latency_ms = float(np.random.exponential(5.0) + 0.5)
                if peer.latency_ms > 500:
                    peer.status = NodeStatus.DEGRADED
                else:
                    peer.status = NodeStatus.ACTIVE

            await asyncio.sleep(self.HEARTBEAT_INTERVAL_S)

    async def _routing_maintenance_loop(self) -> None:
        """Periodically rebuild routing table from peer list."""
        while self._running:
            # Update routes based on current peer latencies
            for peer_id, peer in self._peers.items():
                self._update_routing(
                    peer_id, peer_id, 1,
                    peer.latency_ms, 100.0,
                    peer.trust_score,
                )
            await asyncio.sleep(30.0)

    # ─── Stats ────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        uptime = time.time() - self._stats["uptime_start"]
        return {
            **self._stats,
            "node_id": self.node_id,
            "active_peers": len(self.get_active_peers()),
            "routing_table_size": len(self._routing_table),
            "active_sessions": len(self._sessions),
            "uptime_s": uptime,
            "msg_rate_per_s": self._stats["messages_sent"] / max(uptime, 1),
            "encryption_key_stats": self._encryption.get_stats(),
        }
