import re
from threading import Lock


class FaissStore:
    _DIM = 256

    def __init__(self) -> None:
        self._lock = Lock()
        self._available = False
        self._index = None
        self._ids: list[str] = []
        self._faiss = None
        self._np = None
        self._init_backend()

    @property
    def available(self) -> bool:
        return self._available

    def _init_backend(self) -> None:
        try:
            import faiss  # type: ignore
            import numpy as np

            self._faiss = faiss
            self._np = np
            self._index = faiss.IndexFlatL2(self._DIM)
            self._available = True
        except Exception:
            self._available = False

    def _embed(self, text: str):
        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        vec = self._np.zeros(self._DIM, dtype="float32")
        for token in tokens:
            vec[hash(token) % self._DIM] += 1.0
        norm = float(self._np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec.reshape(1, -1)

    def add_transcript(self, call_id: str, transcript: str) -> bool:
        if not self._available or not transcript.strip():
            return False

        with self._lock:
            vec_np = self._embed(transcript)
            self._index.add(vec_np)
            self._ids.append(call_id)
        return True

    def similar_calls(self, text: str, top_k: int = 5) -> list[str]:
        if not self._available or self._index is None or self._index.ntotal == 0:
            return []

        with self._lock:
            query_np = self._embed(text)
            _, indices = self._index.search(query_np, min(top_k, self._index.ntotal))

        out: list[str] = []
        for idx in indices[0].tolist():
            if 0 <= idx < len(self._ids):
                out.append(self._ids[idx])
        return out
