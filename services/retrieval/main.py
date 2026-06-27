"""
Retrieval Service — FastAPI application.
Generates candidates using the Two-Tower model and FAISS.
"""

import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .faiss_index import FAISSIndex
from .two_tower import TwoTowerInference

logger = logging.getLogger(__name__)

# Global instances
two_tower = None
faiss_index = None

# Stats
stats = {
    "queries": 0,
    "index_builds": 0,
    "total_latency_ms": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager."""
    global two_tower, faiss_index
    logger.info("Starting Retrieval Service...")
    
    # Initialize model
    two_tower = TwoTowerInference()  # Will use random weights until trained
    
    # Initialize empty index (in production, we'd load from disk)
    faiss_index = FAISSIndex()
    
    yield
    logger.info("Shutting down Retrieval Service...")


app = FastAPI(
    title="ReelMind Retrieval Service",
    description="Two-Tower + FAISS candidate generation",
    version="0.1.0",
    lifespan=lifespan,
)


class RetrievalRequest(BaseModel):
    user_id: str
    user_features: list[float]  # 13-dimensional vector from feature engine
    num_candidates: int = 100
    exclude_video_ids: list[str] = []


class RetrievalResponse(BaseModel):
    candidates: list[dict]
    latency_ms: float
    source: str = "two_tower_faiss"


class IndexRebuildRequest(BaseModel):
    video_ids: list[str]
    video_features: list[list[float]]  # list of 17-dim vectors


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "retrieval",
        "index_size": len(faiss_index.video_ids) if faiss_index else 0,
        "stats": stats,
    }


@app.post("/v1/retrieve", response_model=RetrievalResponse)
async def retrieve_candidates(req: RetrievalRequest):
    """Retrieve top-K candidates for a user."""
    start_time = time.time()
    
    if not faiss_index or not faiss_index.video_ids:
        # Fallback if index is empty
        logger.warning("FAISS index is empty. Returning empty list.")
        return RetrievalResponse(candidates=[], latency_ms=0.0)

    try:
        # 1. Generate user embedding
        user_emb = two_tower.get_user_embedding(req.user_features)
        query_np = np.array([user_emb], dtype=np.float32)
        
        # 2. Search FAISS (fetch more than requested in case we need to filter)
        fetch_k = req.num_candidates + len(req.exclude_video_ids)
        raw_results = faiss_index.search(query_np, top_k=fetch_k)
        
        # 3. Filter and format
        exclude_set = set(req.exclude_video_ids)
        candidates = []
        for vid, score in raw_results:
            if vid not in exclude_set:
                candidates.append({"video_id": vid, "score": score})
                if len(candidates) >= req.num_candidates:
                    break
                    
        elapsed_ms = (time.time() - start_time) * 1000
        stats["queries"] += 1
        stats["total_latency_ms"] += elapsed_ms
        
        return RetrievalResponse(
            candidates=candidates,
            latency_ms=round(elapsed_ms, 2)
        )
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/index/rebuild")
async def rebuild_index(req: IndexRebuildRequest):
    """Rebuild the FAISS index with new video features."""
    start_time = time.time()
    
    if len(req.video_ids) != len(req.video_features):
        raise HTTPException(status_code=400, detail="Mismatched IDs and features length")
        
    try:
        # Generate video embeddings in batches
        embeddings = []
        for v_feat in req.video_features:
            emb = two_tower.get_video_embedding(v_feat)
            embeddings.append(emb)
            
        emb_np = np.array(embeddings, dtype=np.float32)
        
        # Build index
        global faiss_index
        new_index = FAISSIndex()
        new_index.build_index(req.video_ids, emb_np, use_pq=False)
        faiss_index = new_index  # Atomic swap
        
        stats["index_builds"] += 1
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            "status": "success",
            "indexed_items": len(req.video_ids),
            "latency_ms": round(elapsed_ms, 2)
        }
    except Exception as e:
        logger.error(f"Index rebuild failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
