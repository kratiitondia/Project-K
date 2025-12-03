import faiss
import numpy as np
from typing import List, Dict, Optional
import os
import pickle

class InMemoryFAISS:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadatas: List[Dict] = []

    def add(self, embeddings: np.ndarray, metadatas: List[Dict]):
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Embeddings must be shape (N, {self.dim})")
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype('float32')
        self.index.add(embeddings)
        self.metadatas.extend(metadatas)

    def search(self, q_emb: np.ndarray, k: int = 5):
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        if q_emb.dtype != np.float32:
            q_emb = q_emb.astype('float32')
        D, I = self.index.search(q_emb, k)
        return D, I

    def get_metadata(self, idx: int) -> Optional[Dict]:
        if idx < 0 or idx >= len(self.metadatas):
            return None
        return self.metadatas[idx]

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "metadatas.pkl"), "wb") as f:
            pickle.dump(self.metadatas, f)

    def load(self, path: str):
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "metadatas.pkl"), "rb") as f:
            self.metadatas = pickle.load(f)
