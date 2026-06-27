"""
RL Optimizer Service — FastAPI application.
Re-ranks final feed using bandits and MMR diversity.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .bandits import BanditReranker, ThompsonSamplingBandit

logger = logging.getLogger(__name__)

# Global instance
reranker = None

stats = {
    "rerank_calls": 0,
    "reward_updates": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global reranker
    logger.info("Starting RL Optimizer Service...")
    reranker = BanditReranker()
    yield
    logger.info("Shutting down RL Optimizer Service...")


app = FastAPI(
    title="ReelMind RL Optimizer Service",
    description="Contextual bandits and diversity re-ranking",
    version="0.1.0",
    lifespan=lifespan,
)


class RerankRequest(BaseModel):
    user_id: str
    user_features: dict
    ranked_candidates: list[dict]  # list of dicts with 'video_id', 'score', 'category'
    num_results: int = 15


class RewardRequest(BaseModel):
    category: str
    reward_value: float  # e.g., watch percentage or binary like


class RerankResponse(BaseModel):
    feed: list[dict]
    latency_ms: float
    policy_version: str = "ts_mmr_v1"


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "rl_optimizer",
        "stats": stats,
    }


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    """
    Apply Bandit exploration and MMR diversity to final candidates.
    """
    start_time = time.time()
    
    try:
        final_feed = reranker.optimize_feed(
            ranked_candidates=req.ranked_candidates,
            user_features=req.user_features,
            num_results=req.num_results
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        stats["rerank_calls"] += 1
        
        return RerankResponse(feed=final_feed, latency_ms=round(elapsed_ms, 2))
        
    except Exception as e:
        logger.error(f"Re-ranking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/reward")
async def process_reward(req: RewardRequest):
    """
    Update the bandit posterior for a category.
    """
    try:
        # Constrain reward
        r = max(0.0, min(req.reward_value, 1.0))
        reranker.ts_bandit.update(req.category, r)
        
        stats["reward_updates"] += 1
        return {"status": "success", "category": req.category}
        
    except Exception as e:
        logger.error(f"Reward update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
