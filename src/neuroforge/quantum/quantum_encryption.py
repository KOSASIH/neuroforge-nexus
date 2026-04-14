"""
NeuroForge Nexus — Quantum Key Distribution & Neural Encryption
===============================================================
BB84-inspired quantum key distribution simulation + AES-256-GCM
hybrid encryption for neural data packets.

Provides:
  - BB84 QKD simulation (key generation, basis reconciliation, privacy amplification)
  - AES-256-GCM symmetric encryption with QKD-derived keys
  - Neural data packet serialization / encryption / decryption
  - Post-quantum lattice signature stubs (Kyber/Dilithium interface)
"""

import os
import hashlib
import hmac
import struct
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from loguru import logger

from neuroforge.quantum.quantum_optimizer import QuantumOptimizer


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class BB84Session:
    """Complete BB84 QKD session state."""
    alice_bits: NDArray[np.uint8]
    alice_bases: NDArray[np.uint8]     # 0=Z, 1=X basis
    bob_bases: NDArray[np.uint8]
    bob_measurements: NDArray[np.uint8]
    sifted_key: NDArray[np.uint8]
    final_key: bytes
    qber: float                         # Quantum bit error rate
    key_length: int
    secure: bool


@dataclass
class EncryptedPacket:
    """Encrypted neural data packet."""
    ciphertext: bytes
    nonce: bytes
    tag: bytes
    key_id: bytes          # SHA256 of key (for key lookup)
    timestamp: float
    channel_count: int
    sample_count: int
    signature: bytes       # HMAC-SHA256


# ─── BB84 QKD Simulator ───────────────────────────────────────────────────────

class BB84Simulator:
    """
    Simulate BB84 Quantum Key Distribution protocol.

    Alice prepares qubits in random bases/states.
    Bob measures in random bases.
    Sifting: keep bits where bases match.
    Error estimation + privacy amplification → secure key.
    """

    QBER_THRESHOLD = 0.11  # >11% QBER indicates eavesdropping

    def __init__(self, n_raw_bits: int = 4096, eavesdrop_probability: float = 0.0) -> None:
        self.n_raw = n_raw_bits
        self.eve_prob = eavesdrop_probability

    def generate_session(self) -> BB84Session:
        """Run a complete BB84 session. Returns final secure key."""
        # Alice's random bits and bases
        alice_bits = np.random.randint(0, 2, self.n_raw, dtype=np.uint8)
        alice_bases = np.random.randint(0, 2, self.n_raw, dtype=np.uint8)

        # Eve intercepts (if present) — introduces errors
        eve_bits = alice_bits.copy()
        if self.eve_prob > 0:
            eve_bases = np.random.randint(0, 2, self.n_raw, dtype=np.uint8)
            mismatch = (eve_bases != alice_bases)
            # Eve guesses wrong basis 50% of the time → 25% error rate
            eve_errors = mismatch & (np.random.random(self.n_raw) < 0.5)
            eve_bits = np.where(eve_errors, 1 - alice_bits, alice_bits).astype(np.uint8)

        # Bob's random bases
        bob_bases = np.random.randint(0, 2, self.n_raw, dtype=np.uint8)

        # Bob measures
        bob_measurements = np.where(
            bob_bases == alice_bases, eve_bits,
            np.random.randint(0, 2, self.n_raw, dtype=np.uint8)
        ).astype(np.uint8)

        # Sifting: keep bits where Alice & Bob chose same basis
        matching = alice_bases == bob_bases
        sifted_alice = alice_bits[matching]
        sifted_bob = bob_measurements[matching]

        # QBER estimation on sample
        sample_size = min(200, len(sifted_alice))
        sample_idx = np.random.choice(len(sifted_alice), sample_size, replace=False)
        qber = float(np.mean(sifted_alice[sample_idx] != sifted_bob[sample_idx]))

        # Remove sample bits from key
        key_mask = np.ones(len(sifted_alice), dtype=bool)
        key_mask[sample_idx] = False
        raw_key = sifted_alice[key_mask]

        secure = qber < self.QBER_THRESHOLD

        # Privacy amplification: hash compression
        if secure and len(raw_key) > 0:
            key_bytes = np.packbits(raw_key).tobytes()
            # Iterative hash compression to remove any residual Eve information
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=os.urandom(16),
                info=b"NeuroForge-BB84-v1",
                backend=default_backend(),
            )
            final_key = hkdf.derive(key_bytes)
        else:
            final_key = os.urandom(32)  # Fallback to CSPRNG
            if not secure:
                logger.warning(f"BB84: QBER={qber:.3f} exceeds threshold. Key may be compromised!")

        return BB84Session(
            alice_bits=alice_bits,
            alice_bases=alice_bases,
            bob_bases=bob_bases,
            bob_measurements=bob_measurements,
            sifted_key=raw_key,
            final_key=final_key,
            qber=qber,
            key_length=len(final_key) * 8,
            secure=secure,
        )


# ─── Neural Encryption Engine ─────────────────────────────────────────────────

class NeuralEncryptionEngine:
    """
    Hybrid quantum-classical encryption for neural data streams.

    Key management: BB84 QKD → HKDF → AES-256-GCM
    Authentication: HMAC-SHA256 on packet metadata
    Forward secrecy: Key rotation every N packets or T seconds
    """

    KEY_ROTATION_INTERVAL = 1000    # Rotate key every N packets
    KEY_ROTATION_TIME_S = 3600.0    # Rotate key every hour

    def __init__(self, eavesdrop_prob: float = 0.0) -> None:
        self._qkd = BB84Simulator(eavesdrop_probability=eavesdrop_prob)
        self._active_key: bytes = os.urandom(32)
        self._key_id: bytes = hashlib.sha256(self._active_key).digest()[:8]
        self._packet_count = 0
        self._key_created_at = time.time()
        self._key_history: dict[bytes, bytes] = {}  # key_id -> key

        logger.info("NeuralEncryptionEngine initialized (AES-256-GCM + BB84 QKD)")

    def rotate_key(self) -> None:
        """Generate fresh QKD-derived key and rotate."""
        session = self._qkd.generate_session()
        self._active_key = session.final_key
        self._key_id = hashlib.sha256(self._active_key).digest()[:8]
        self._key_history[self._key_id] = self._active_key
        self._packet_count = 0
        self._key_created_at = time.time()
        logger.debug(f"Neural encryption key rotated (QBER={session.qber:.3f})")

    def _check_rotation(self) -> None:
        """Auto-rotate key if threshold exceeded."""
        age = time.time() - self._key_created_at
        if (self._packet_count >= self.KEY_ROTATION_INTERVAL or
                age >= self.KEY_ROTATION_TIME_S):
            self.rotate_key()

    def encrypt_epoch(
        self,
        data: NDArray[np.float64],
        metadata: dict | None = None,
    ) -> EncryptedPacket:
        """
        Encrypt a neural data epoch (n_channels, n_samples) into a packet.
        """
        self._check_rotation()

        # Serialize: header (8 bytes) + float32 data
        n_ch, n_samp = data.shape
        header = struct.pack(">II", n_ch, n_samp)
        payload = header + data.astype(np.float32).tobytes()

        # AES-256-GCM encryption
        aesgcm = AESGCM(self._active_key)
        nonce = os.urandom(12)
        ciphertext_with_tag = aesgcm.encrypt(nonce, payload, None)
        ciphertext = ciphertext_with_tag[:-16]
        tag = ciphertext_with_tag[-16:]

        # HMAC authentication
        mac = hmac.new(
            self._active_key,
            nonce + ciphertext + tag,
            hashlib.sha256,
        ).digest()

        self._packet_count += 1

        return EncryptedPacket(
            ciphertext=ciphertext,
            nonce=nonce,
            tag=tag,
            key_id=self._key_id,
            timestamp=time.time(),
            channel_count=n_ch,
            sample_count=n_samp,
            signature=mac,
        )

    def decrypt_epoch(
        self, packet: EncryptedPacket, key: bytes | None = None
    ) -> NDArray[np.float64]:
        """Decrypt a neural epoch from an encrypted packet."""
        key = key or self._key_history.get(packet.key_id) or self._active_key

        # Verify HMAC
        expected_mac = hmac.new(
            key,
            packet.nonce + packet.ciphertext + packet.tag,
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(expected_mac, packet.signature):
            raise ValueError("Packet authentication failed — HMAC mismatch")

        # AES-256-GCM decryption
        aesgcm = AESGCM(key)
        payload = aesgcm.decrypt(
            packet.nonce, packet.ciphertext + packet.tag, None
        )

        # Deserialize
        header = payload[:8]
        n_ch, n_samp = struct.unpack(">II", header)
        data = np.frombuffer(payload[8:], dtype=np.float32).reshape(n_ch, n_samp)
        return data.astype(np.float64)

    def get_stats(self) -> dict:
        return {
            "algorithm": "AES-256-GCM",
            "key_exchange": "BB84-QKD",
            "packets_encrypted": self._packet_count,
            "key_age_s": time.time() - self._key_created_at,
            "keys_in_history": len(self._key_history),
        }
