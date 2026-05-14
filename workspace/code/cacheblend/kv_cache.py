"""In-memory per-(chunk_hash, layer_id) KV cache store.

Hashing follows vLLM's sha256_cbor convention: SHA-256 over a CBOR-serialized
tuple of (parent_hash, list_of_token_ids). Falls back to JSON encoding when
``cbor2`` is unavailable so unit tests can run in a minimal environment.

Adapted from: https://github.com/LMCache/LMCache (lmcache/v1/token_database.py
:_load_hash_function) and https://github.com/YaoJiayi/CacheBlend.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable, Optional, Tuple

import torch

try:
    import cbor2  # type: ignore
    _HAS_CBOR = True
except ImportError:  # pragma: no cover - tested via env without cbor2
    _HAS_CBOR = False


def _serialize_block(parent_hash: Optional[bytes], token_ids: Iterable[int]) -> bytes:
    """Deterministic byte encoding of (parent_hash, token_id_block)."""
    token_list = list(int(t) for t in token_ids)
    if _HAS_CBOR:
        return cbor2.dumps([parent_hash if parent_hash is not None else b"", token_list])
    payload = {
        "parent": (parent_hash or b"").hex(),
        "tokens": token_list,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_cbor(token_ids: Iterable[int], parent_hash: Optional[bytes] = None) -> bytes:
    """SHA-256 hash of the CBOR encoding of (parent_hash, token_ids)."""
    return hashlib.sha256(_serialize_block(parent_hash, token_ids)).digest()


class ChunkKVStore:
    """CPU-resident KV cache, keyed by (chunk_hash, layer_id).

    Tensors are stored in their native dtype (typically the model's activation
    dtype, e.g. fp16). K is expected to be in pre-RoPE form; rotation is
    re-applied at fetch time by ``cacheblend.blend.recover_rope_k``.
    """

    def __init__(self, num_layers: int, dtype: torch.dtype, device: str = "cpu") -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.num_layers = int(num_layers)
        self.dtype = dtype
        self.device = device
        # _store[chunk_hash] is a list of length num_layers; each entry is None
        # or a (K, V) tuple.
        self._store: dict[bytes, list[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = {}

    # ------------------------------------------------------------------ hash
    @staticmethod
    def hash_chunk(token_ids: Iterable[int], parent_hash: Optional[bytes] = None) -> bytes:
        return sha256_cbor(token_ids, parent_hash)

    # ------------------------------------------------------------------ put
    def put(
        self,
        chunk_hash: bytes,
        layer_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        if not 0 <= layer_id < self.num_layers:
            raise IndexError(f"layer_id {layer_id} out of range [0,{self.num_layers})")
        if chunk_hash not in self._store:
            self._store[chunk_hash] = [None] * self.num_layers
        k_store = k.detach().to(self.device, dtype=self.dtype).contiguous().clone()
        v_store = v.detach().to(self.device, dtype=self.dtype).contiguous().clone()
        self._store[chunk_hash][layer_id] = (k_store, v_store)

    # ---------------------------------------------------------------- fetch
    def fetch(
        self, chunk_hash: bytes, layer_id: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not 0 <= layer_id < self.num_layers:
            raise IndexError(f"layer_id {layer_id} out of range [0,{self.num_layers})")
        layers = self._store.get(chunk_hash)
        if layers is None:
            return None
        return layers[layer_id]

    def has(self, chunk_hash: bytes) -> bool:
        return chunk_hash in self._store

    def __len__(self) -> int:
        return len(self._store)
