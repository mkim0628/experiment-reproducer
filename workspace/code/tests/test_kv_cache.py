import sys
import pathlib

# Make workspace/code/ importable when running pytest from repo root.
_HERE = pathlib.Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import torch

from cacheblend.kv_cache import ChunkKVStore, sha256_cbor


def test_hash_is_deterministic() -> None:
    h1 = sha256_cbor([1, 2, 3, 4])
    h2 = sha256_cbor([1, 2, 3, 4])
    assert h1 == h2
    assert isinstance(h1, bytes) and len(h1) == 32
    # Different parent hash flips the output.
    h_with_parent = sha256_cbor([1, 2, 3, 4], parent_hash=b"\x00" * 32)
    assert h1 != h_with_parent
    # Different tokens differ.
    assert sha256_cbor([1, 2, 3]) != h1


def test_put_fetch_roundtrip() -> None:
    store = ChunkKVStore(num_layers=4, dtype=torch.float32, device="cpu")
    chunk_hash = ChunkKVStore.hash_chunk([10, 20, 30])
    k = torch.randn(8, 7, 16)  # (num_kv_heads, chunk_len, head_dim)
    v = torch.randn(8, 7, 16)
    store.put(chunk_hash, 2, k, v)
    out = store.fetch(chunk_hash, 2)
    assert out is not None
    k_out, v_out = out
    assert k_out.shape == k.shape
    assert v_out.shape == v.shape
    assert torch.equal(k_out, k)
    assert torch.equal(v_out, v)


def test_miss_returns_none() -> None:
    store = ChunkKVStore(num_layers=2, dtype=torch.float32)
    assert store.fetch(b"\x11" * 32, 0) is None
    assert not store.has(b"\x11" * 32)


def test_dtype_preserved() -> None:
    store = ChunkKVStore(num_layers=1, dtype=torch.float16)
    k = torch.randn(4, 3, 8, dtype=torch.float32)
    v = torch.randn(4, 3, 8, dtype=torch.float32)
    h = ChunkKVStore.hash_chunk([7, 8, 9])
    store.put(h, 0, k, v)
    k_out, v_out = store.fetch(h, 0)
    assert k_out.dtype == torch.float16
    assert v_out.dtype == torch.float16
