import faiss, numpy as np
from typing import Iterable, Dict, Any, List, Tuple

class FaissDedupWrapper:
    """
    A tiny wrapper that keeps an ANN index (HNSW) and collapses near-duplicates on insert.
    Assumes embeddings are L2-normalized so dot == cosine.
    """
    def __init__(self, dim: int, m: int = 32, ef_search: int = 64, ef_construction: int = 200,
                 dup_threshold: float = 0.94):
        self.dim = dim
        self.dup_threshold = dup_threshold

        # HNSW is great for incremental insert/search
        self.index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

        # Map from FAISS internal order → your row id
        self.kept_ids: List[int] = []            # original row ids kept
        self.rep_to_dups: Dict[int, List[int]] = {}  # {rep_row_id: [dup_row_ids...]}

    def _nearest(self, v: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self.index.ntotal == 0:
            return np.array([[]], dtype="float32"), np.array([[]], dtype="int64")
        D, I = self.index.search(v[None, :].astype("float32"), k)
        return D, I

    def add_with_dedup(self, vec: np.ndarray, row_id: int, k_probe: int = 5) -> bool:
        """
        Returns True if kept, False if treated as duplicate.
        """
        if self.index.ntotal == 0:
            self.index.add(vec[None, :].astype("float32"))
            self.kept_ids.append(row_id)
            self.rep_to_dups.setdefault(row_id, [])
            return True

        D, I = self._nearest(vec, k=k_probe)
        best_sim, best_internal = float(D[0, 0]), int(I[0, 0])
        if best_internal >= 0 and best_sim >= self.dup_threshold:
            # Map internal index -> kept row id
            rep_row_id = self.kept_ids[best_internal]
            self.rep_to_dups.setdefault(rep_row_id, []).append(row_id)
            return False

        # Keep it
        self.index.add(vec[None, :].astype("float32"))
        self.kept_ids.append(row_id)
        self.rep_to_dups.setdefault(row_id, [])
        return True

    def add_many_with_dedup(self, vectors: np.ndarray, row_ids: Iterable[int], k_probe: int = 5) -> Tuple[List[int], Dict[int, List[int]]]:
        kept = []
        for v, rid in zip(vectors, row_ids):
            if self.add_with_dedup(v, rid, k_probe=k_probe):
                kept.append(rid)
        return kept, self.rep_to_dups
