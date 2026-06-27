"""
Ranking Service — FastAPI application.
Provides two-stage ranking: LightGBM Pre-Ranker and DeepFM Heavy Ranker.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .deepfm import DeepFMRanker
from .preranker import LightGBMPreRanker

logger = logging.getLogger(__name__)

# Global instances
preranker = None
deepfm = None

stats = {
    "prerank_calls": 0,
    "heavy_rank_calls": 0,
    "total_videos_preranked": 0,
    "total_videos_heavy_ranked": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global preranker, deepfm
    logger.info("Starting Ranking Service...")
    
    preranker = LightGBMPreRanker()
    deepfm = DeepFMRanker()
    
    yield
    logger.info("Shutting down Ranking Service...")


app = FastAPI(
    title="ReelMind Ranking Service",
    description="Two-stage ranking pipeline",
    version="0.1.0",
    lifespan=lifespan,
)


class RankingRequest(BaseModel):
    user_id: str
    user_features: list[float]
    candidate_ids: list[str]
    video_features_list: list[list[float]]
    cross_features_list: list[list[float]]
    top_k: int = 50


class RankingResponse(BaseModel):
    ranked_candidates: list[dict]
    latency_ms: float
    model_version: str = "v1"


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ranking",
        "stats": stats,
    }


@app.post("/v1/prerank", response_model=RankingResponse)
async def prerank(req: RankingRequest):
    """
    Fast first-stage ranking using LightGBM.
    Filters hundreds of retrieval candidates down to top-K.
    """
    start_time = time.time()
    
    if not len(req.candidate_ids) == len(req.video_features_list) == len(req.cross_features_list):
        raise HTTPException(status_code=400, detail="Mismatched input lists length")
        
    try:
        ranked_tuples = preranker.rerank(
            candidate_ids=req.candidate_ids,
            user_features=req.user_features,
            video_features_list=req.video_features_list,
            cross_features_list=req.cross_features_list,
            top_k=req.top_k
        )
        
        candidates = [{"video_id": vid, "score": score} for vid, score in ranked_tuples]
        
        elapsed_ms = (time.time() - start_time) * 1000
        stats["prerank_calls"] += 1
        stats["total_videos_preranked"] += len(req.candidate_ids)
        
        return RankingResponse(ranked_candidates=candidates, latency_ms=round(elapsed_ms, 2))
        
    except Exception as e:
        logger.error(f"Pre-ranking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/rank", response_model=RankingResponse)
async def rank(req: RankingRequest):
    """
    Heavy second-stage ranking using DeepFM.
    Accurately scores the top candidates from the pre-ranker.
    """
    start_time = time.time()
    
    if not len(req.candidate_ids) == len(req.video_features_list) == len(req.cross_features_list):
        raise HTTPException(status_code=400, detail="Mismatched input lists length")
        
    try:
        # In a real system, we'd also request more complex features here
        ranked_tuples = deepfm.rank(
            candidate_ids=req.candidate_ids,
            user_features=req.user_features,
            video_features_list=req.video_features_list,
            cross_features_list=req.cross_features_list
        )
        
        candidates = [{"video_id": vid, "score": score} for vid, score in ranked_tuples[:req.top_k]]
        
        elapsed_ms = (time.time() - start_time) * 1000
        stats["heavy_rank_calls"] += 1
        stats["total_videos_heavy_ranked"] += len(req.candidate_ids)
        
        return RankingResponse(ranked_candidates=candidates, latency_ms=round(elapsed_ms, 2))
        
    except Exception as e:
        logger.error(f"Heavy ranking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
