from fastapi import APIRouter, HTTPException
from .schemas import ChatRequest, ChatResponse
from .pii_masker import PIIMasker
from .context_enricher import ContextEnricher
from .datastore import InMemoryFAISS
from .embeddings_openai import embed_texts
from .rag import RAG
from .llm import call_chat_system
from typing import List, Dict, Any
import numpy as np

router = APIRouter()

# Initialize vectorstore by embedding a sample to discover dim
_sample_emb = embed_texts(["init"])
DIM = _sample_emb.shape[1]
VECTORSTORE = InMemoryFAISS(dim=DIM)
RAG_PIPE = RAG(VECTORSTORE)

@router.post("/index")
async def index_documents(docs: List[Dict[str, str]]):
    texts = [d["text"] for d in docs]
    embeddings = embed_texts(texts)
    metadatas = [{"id": d.get("id"), "text": d.get("text"), "title": d.get("title")} for d in docs]
    VECTORSTORE.add(embeddings, metadatas)
    return {"indexed": len(docs)}

@router.post("/seed")
async def seed_route():
    from .seed_data import seed
    n = seed(VECTORSTORE)
    return {"seeded": n}

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    clean_message = PIIMasker.mask(req.message)
    ctx = ContextEnricher.enrich(req.user_id, req.latitude, req.longitude)
    retrieved = RAG_PIPE.retrieve(clean_message, k=4)

    system_prompt = (
        "You are a helpful customer support assistant. Answer using only the provided sources. "
        "If not present, say 'I don't know yet; let me check' and provide safe next steps."
    )

    sources_texts = ""
    for i, s in enumerate(retrieved):
        sources_texts += f"Source {i+1} Title: {s['metadata'].get('title','')}\n"
        sources_texts += f"Source {i+1} Text: {s['metadata'].get('text','')}\n\n"

    user_prompt = f"User message: {clean_message}\n\nContext: {ctx}\n\nSources:\n{sources_texts}\nProvide a concise answer and list the sources used."

    try:
        reply = call_chat_system(system_prompt, user_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(reply=reply, sources=[s["metadata"] for s in retrieved], meta={"masked_message": clean_message})
