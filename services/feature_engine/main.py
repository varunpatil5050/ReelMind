"""
ReelMind Feature Engine Service — FastAPI application.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .feature_store import FeatureStore, UserFeatures, VideoFeatures, CrossFeatures

logger = logging.getLogger(__name__)

# Config
REDIS_URL = "redis://redis:6379"
POSTGRES_DSN = "postgresql://reelmind:reelmind_dev@postgres:5432/reelmind"

feature_store = FeatureStore(redis_url=REDIS_URL, postgres_dsn=POSTGRES_DSN)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to initialize and close connections."""
    logger.info("Starting Feature Engine Service...")
    await feature_store.initialize()
    yield
    logger.info("Shutting down Feature Engine Service...")
    await feature_store.close()


app = FastAPI(
    title="ReelMind Feature Engine",
    description="Online feature computation service",
    version="0.1.0",
    lifespan=lifespan,
)


# ─── Request Models ──────────────────────────────────────────────────────────


class UserFeatureRequest(BaseModel):
    user_id: str


class VideoFeatureRequest(BaseModel):
    video_id: str


class BatchUserFeatureRequest(BaseModel):
    user_ids: list[str]


class BatchVideoFeatureRequest(BaseModel):
    video_ids: list[str]


class CrossFeatureRequest(BaseModel):
    user_id: str
    video_id: str


class BatchCrossFeatureRequest(BaseModel):
    user_id: str
    video_ids: list[str]


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "feature_engine",
        "cache_stats": feature_store.cache_stats,
    }


@app.post("/v1/features/user")
async def get_user_features(req: UserFeatureRequest):
    features = await feature_store.get_user_features(req.user_id)
    return {"features": features.to_dict(), "vector": features.to_vector().tolist()}


@app.post("/v1/features/user/batch")
async def get_user_features_batch(req: BatchUserFeatureRequest):
    results = await feature_store.get_user_features_batch(req.user_ids)
    return {
        uid: {"features": f.to_dict(), "vector": f.to_vector().tolist()}
        for uid, f in results.items()
    }


@app.post("/v1/features/video")
async def get_video_features(req: VideoFeatureRequest):
    features = await feature_store.get_video_features(req.video_id)
    return {"features": features.to_dict(), "vector": features.to_vector().tolist()}


@app.post("/v1/features/video/batch")
async def get_video_features_batch(req: BatchVideoFeatureRequest):
    results = await feature_store.get_video_features_batch(req.video_ids)
    return {
        vid: {"features": f.to_dict(), "vector": f.to_vector().tolist()}
        for vid, f in results.items()
    }


@app.post("/v1/features/cross")
async def get_cross_features(req: CrossFeatureRequest):
    uf = await feature_store.get_user_features(req.user_id)
    vf = await feature_store.get_video_features(req.video_id)
    cf = await feature_store.compute_cross_features(uf, vf)
    return {
        "user_id": req.user_id,
        "video_id": req.video_id,
        "vector": cf.to_vector().tolist(),
    }


@app.post("/v1/features/cross/batch")
async def get_cross_features_batch(req: BatchCrossFeatureRequest):
    """Efficiently compute cross features for a single user and many videos."""
    uf = await feature_store.get_user_features(req.user_id)
    vfs = await feature_store.get_video_features_batch(req.video_ids)
    
    results = {}
    for vid, vf in vfs.items():
        cf = await feature_store.compute_cross_features(uf, vf)
        results[vid] = cf.to_vector().tolist()
        
    return {"user_id": req.user_id, "cross_features": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
