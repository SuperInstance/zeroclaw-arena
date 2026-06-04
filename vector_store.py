"""
VectorStore — ZeroClaw wrapper around torch-vector-search.

Drop-in replacement for the SQLite-based VectorDB that ships with zeroclaw.py.
Uses the GPU/CPU auto-selecting VectorIndex from torch-vector-search for
significantly faster similarity search at scale.

Interface matches the original VectorDB:
    - insert(id, text, metadata)
    - search(query_text, top_k) -> list[tuple[str, float, dict]]
    - count() -> int
    - close()
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from torch_vector_search import VectorIndex
from torch_vector_search.embedders import HashEmbedder


class VectorStore:
    """GPU-accelerated vector store backed by torch-vector-search.

    Same external interface as VectorDB so ZeroClaw can swap transparently.
    Internally uses a deterministic hash-based embedder so that identical
    text always maps to the same vector, matching the original semantics.
    """

    _DIM = 384  # use a richer embedding space than the original 64-dim hash

    def __init__(self, path: str):
        self._path = Path(path)
        self._embedder = HashEmbedder(dim=self._DIM)
        self._index = VectorIndex(dim=self._DIM)
        # Track id -> internal index mapping for lookup
        self._id_to_idx: dict[str, int] = {}
        self._ids: list[str] = []
        self._persist_dir = self._path.parent / "torch_vectors"

    # -- embedding (deterministic, same text = same vector) ----------------

    def _embed(self, text: str) -> np.ndarray:
        return self._embedder.embed(text)

    # -- public API (matches VectorDB) -------------------------------------

    def insert(self, id: str, text: str, metadata: dict) -> None:
        """Insert or update a vector entry."""
        vec = self._embed(text).reshape(1, -1)
        if id in self._id_to_idx:
            # Update: remove old, add new
            old_idx = self._id_to_idx[id]
            self._index.remove(old_idx)
            # Rebuild id mapping (indices shift after remove)
            self._rebuild_id_map_after_remove(old_idx)

        meta = dict(metadata)
        meta["_id"] = id
        meta["_text"] = text
        self._index.add(vec, metadata=[meta])
        new_idx = self._index.count - 1
        self._id_to_idx[id] = new_idx
        self._ids.append(id)

    def search(self, query_text: str, top_k: int = 10) -> list[tuple[str, float, dict]]:
        """Search for similar entries. Returns [(id, score, metadata), ...]."""
        if self._index.count == 0:
            return []
        query_vec = self._embed(query_text).reshape(1, -1)
        results = self._index.search(query_vec, top_k=top_k)
        out = []
        for r in results:
            meta = dict(r.metadata)  # copy
            entry_id = meta.pop("_id", str(r.index))
            # Pop internal text field from metadata
            meta.pop("_text", None)
            out.append((entry_id, r.score, meta))
        return out

    def count(self) -> int:
        return self._index.count

    def close(self) -> None:
        """Persist index to disk."""
        try:
            self._index.save(self._persist_dir)
        except Exception:
            pass  # best-effort persist

    # -- internal helpers ---------------------------------------------------

    def _rebuild_id_map_after_remove(self, removed_idx: int) -> None:
        """After removing an entry, rebuild the id-to-index mapping."""
        new_map: dict[str, int] = {}
        new_ids: list[str] = []
        # Re-scan remaining metadata to rebuild mapping
        for i in range(self._index.count):
            meta = self._index._metadata[i]
            eid = meta.get("_id", None)
            if eid:
                new_map[eid] = i
                new_ids.append(eid)
        self._id_to_idx = new_map
        self._ids = new_ids
