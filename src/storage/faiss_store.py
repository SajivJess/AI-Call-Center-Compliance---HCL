from threading import Lock


class FaissStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._available = False
        self._index = None
        self._ids: list[str] = []
        self._model = None
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
            from sentence_transformers import SentenceTransformer

            self._faiss = faiss
            self._np = np
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._index = faiss.IndexFlatL2(384)
            self._available = True
        except Exception:
            self._available = False

    def add_transcript(self, call_id: str, transcript: str) -> bool:
        if not self._available or not transcript.strip():
            return False

        with self._lock:
            vector = self._model.encode([transcript], normalize_embeddings=True)
            vec_np = self._np.asarray(vector, dtype="float32")
            self._index.add(vec_np)
            self._ids.append(call_id)
        return True

    def similar_calls(self, text: str, top_k: int = 5) -> list[str]:
        if not self._available or self._index is None or self._index.ntotal == 0:
            return []

        with self._lock:
            query = self._model.encode([text], normalize_embeddings=True)
            query_np = self._np.asarray(query, dtype="float32")
            _, indices = self._index.search(query_np, min(top_k, self._index.ntotal))

        out: list[str] = []
        for idx in indices[0].tolist():
            if 0 <= idx < len(self._ids):
                out.append(self._ids[idx])
        return out
