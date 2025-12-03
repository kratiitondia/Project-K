import os
import openai
from typing import List
import numpy as np

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")
openai.api_key = OPENAI_API_KEY

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.empty((0,0), dtype="float32")
    resp = openai.Embedding.create(model=EMBED_MODEL, input=texts)
    embeddings = [r["embedding"] for r in resp["data"]]
    arr = np.array(embeddings, dtype="float32")
    return arr
