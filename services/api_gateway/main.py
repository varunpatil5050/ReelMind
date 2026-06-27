"""
ReelMind API Gateway — Primary entry point for all recommendation requests.

Responsibility: Orchestrate pipeline: Feature Fetch → Retrieval → Ranking → RL Optimizer.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import os
import random as stdlib_random

from .session_store import session_store

from .service_client import (
    FeatureEngineClient,
    RetrievalClient,
    RankingClient,
    RLOptimizerClient
)

logger = logging.getLogger(__name__)

# Global Clients
feature_client = None
retrieval_client = None
ranking_client = None
rl_client = None

# Using localhost for local dev, in prod use k8s service names
FE_URL = "http://localhost:8002"
RETRIEVAL_URL = "http://localhost:8003"
RANKING_URL = "http://localhost:8004"
RL_URL = "http://localhost:8005"

# ─── Prometheus Metrics ──────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "reelmind_gateway_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "reelmind_gateway_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
)
FEED_LATENCY = Histogram(
    "reelmind_feed_latency_seconds",
    "End-to-end feed generation latency",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global feature_client, retrieval_client, ranking_client, rl_client
    logger.info("Starting API Gateway...")
    
    feature_client = FeatureEngineClient(FE_URL)
    retrieval_client = RetrievalClient(RETRIEVAL_URL)
    ranking_client = RankingClient(RANKING_URL)
    rl_client = RLOptimizerClient(RL_URL)
    
    yield
    
    logger.info("Shutting down API Gateway...")
    await feature_client.close()
    await retrieval_client.close()
    await ranking_client.close()
    await rl_client.close()


app = FastAPI(
    title="ReelMind API Gateway",
    description="Orchestrator for the recommendation microservices.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start

    endpoint = request.url.path
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)

    response.headers["X-Request-Duration-Ms"] = f"{elapsed * 1000:.2f}"
    return response


# ─── Request Models ──────────────────────────────────────────────────────────

class FeedRequest(BaseModel):
    user_id: str
    num_results: int = Field(ge=1, le=50, default=15)
    session_id: str | None = None
    exclude_video_ids: list[str] = Field(default_factory=list)


class EventRequest(BaseModel):
    user_id: str
    video_id: str
    event_type: str  # like, skip, watch, share, comment
    category: str = "unknown"
    watch_duration_ms: float = 0
    watch_percentage: float = 0


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "api_gateway"}

# Mount frontend
frontend_path = os.path.join(os.path.dirname(__file__), "../../frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(frontend_path, "index.html"))

@app.get("/analytics")
async def serve_analytics():
    return FileResponse(os.path.join(frontend_path, "analytics.html"))

@app.get("/dashboard")
async def serve_dashboard():
    return FileResponse(os.path.join(frontend_path, "dashboard.html"))

@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/v1/feed")
async def get_feed(request: FeedRequest):
    """
    Main endpoint for the client to get a personalized video feed.
    """
    start_total = time.time()
    request_id = str(uuid.uuid4())
    timings = {}
    
    try:
        # STEP 1: Fetch User Features
        t0 = time.time()
        user_data = await feature_client.get_user_features(request.user_id)
        u_vector = user_data["vector"]
        timings["user_features_ms"] = round((time.time() - t0) * 1000, 2)
        
        # STEP 2: Candidate Retrieval
        t0 = time.time()
        retrieval_data = await retrieval_client.get_candidates(request.user_id, u_vector, num_candidates=200)
        candidates = retrieval_data["candidates"]
        timings["retrieval_ms"] = round((time.time() - t0) * 1000, 2)
        
        if not candidates:
            return {"request_id": request_id, "feed": [], "debug": "No candidates found"}
            
        candidate_ids = [c["video_id"] for c in candidates]
        
        # STEP 3: Fetch Video & Cross Features for Ranking
        t0 = time.time()
        video_features_data = await feature_client.get_video_features_batch(candidate_ids)
        cross_features_data = await feature_client.get_cross_features_batch(request.user_id, candidate_ids)
        
        v_features_list = []
        c_features_list = []
        for vid in candidate_ids:
            v_features_list.append(video_features_data.get(vid, {}).get("vector", [0.0]*17))
            c_features_list.append(cross_features_data["cross_features"].get(vid, [0.0]*8))
        timings["item_features_ms"] = round((time.time() - t0) * 1000, 2)
        
        # STEP 4: First-Stage Ranking
        t0 = time.time()
        rank_payload = {
            "user_id": request.user_id,
            "user_features": u_vector,
            "candidate_ids": candidate_ids,
            "video_features_list": v_features_list,
            "cross_features_list": c_features_list,
            "top_k": 50
        }
        prerank_data = await ranking_client.prerank(rank_payload)
        preranked_candidates = prerank_data["ranked_candidates"]
        timings["prerank_ms"] = round((time.time() - t0) * 1000, 2)
        
        # STEP 5: Second-Stage Ranking
        t0 = time.time()
        pr_ids = [c["video_id"] for c in preranked_candidates]
        pr_v_feats = []
        pr_c_feats = []
        for vid in pr_ids:
            pr_v_feats.append(video_features_data.get(vid, {}).get("vector", [0.0]*17))
            pr_c_feats.append(cross_features_data["cross_features"].get(vid, [0.0]*8))
            
        heavy_rank_payload = {
            "user_id": request.user_id,
            "user_features": u_vector,
            "candidate_ids": pr_ids,
            "video_features_list": pr_v_feats,
            "cross_features_list": pr_c_feats,
            "top_k": 30
        }
        heavy_rank_data = await ranking_client.rank(heavy_rank_payload)
        ranked_candidates = heavy_rank_data["ranked_candidates"]
        timings["heavy_rank_ms"] = round((time.time() - t0) * 1000, 2)
        
        # STEP 6: RL Optimizer
        t0 = time.time()
        import random
        cats = ["comedy", "gaming", "education", "music", "vlog"]
        for c in ranked_candidates:
            c["category"] = random.choice(cats)
            
        rl_payload = {
            "user_id": request.user_id,
            "user_features": {"id": request.user_id},
            "ranked_candidates": ranked_candidates,
            "num_results": request.num_results
        }
        rl_data = await rl_client.optimize(rl_payload)
        final_feed = rl_data["feed"]
        timings["rl_ms"] = round((time.time() - t0) * 1000, 2)
        
        elapsed_ms = (time.time() - start_total) * 1000
        timings["total_ms"] = round(elapsed_ms, 2)
        FEED_LATENCY.observe(elapsed_ms / 1000)
        
        # Map to ScoredVideoResponse format with explanation metadata
        cats = ["comedy", "gaming", "education", "music", "vlog", "tech", "sports", "food"]
        final_videos = []
        for i, c in enumerate(final_feed):
            video_cat = c.get("category", stdlib_random.choice(cats))
            final_videos.append({
                "video_id": c["video_id"],
                "score": c.get("rl_score", c.get("score", 0.0)),
                "rank": i + 1,
                "predicted_watch_pct": c.get("score", 0.0),
                "retrieval_source": "two_tower",
                "category": video_cat,
                "exploration": c.get("exploration", False),
                "diversity_penalty": c.get("diversity_penalty", 0.0),
            })
        
        # Store in session for analytics
        session = session_store.get_or_create(request.user_id)
        session.last_feed = final_videos
        session.last_feed_timings = timings
        import time as _t
        session.last_feed_timestamp = _t.time()
            
        return {
            "request_id": request_id,
            "user_id": request.user_id,
            "videos": final_videos,
            "total_latency_ms": timings["total_ms"],
            "model_version": "v1.0",
            "debug_timings": timings
        }
        
    except Exception as e:
        logger.error(f"Feed generation failed: {e}")
        raise HTTPException(status_code=500, detail="Recommendation pipeline failed.")

# ─── Event Collection & Feedback Loop ────────────────────────────────────────


@app.post("/v1/events")
async def collect_event(event: EventRequest):
    """
    Receive user interaction events from the frontend.
    Updates session tracking and RL bandit posteriors.
    """
    # Track in session store
    session = session_store.get_or_create(event.user_id)
    session.record_event(
        event_type=event.event_type,
        video_id=event.video_id,
        category=event.category,
        watch_duration_ms=event.watch_duration_ms,
        watch_percentage=event.watch_percentage,
    )
    
    # Forward reward to RL optimizer (feedback loop)
    reward = 0.0
    if event.event_type == "like":
        reward = 1.0
    elif event.event_type == "share":
        reward = 0.9
    elif event.event_type == "comment":
        reward = 0.7
    elif event.event_type == "watch" and event.watch_percentage > 0.8:
        reward = 0.6
    elif event.event_type == "skip":
        reward = 0.05
    
    if reward > 0 and event.category != "unknown":
        try:
            await rl_client.client.post(
                f"{RL_URL}/v1/reward",
                json={"category": event.category, "reward_value": reward}
            )
        except Exception:
            pass  # Non-critical: don't fail the event collection
    
    return {"status": "ok", "session_watches": session.total_watches}


@app.get("/v1/user/{user_id}/profile")
async def get_user_profile(user_id: str):
    """
    Return user session profile for the analytics dashboard.
    Includes category affinities, session stats, and last feed.
    """
    session = session_store.get(user_id)
    if not session:
        return {
            "user_id": user_id,
            "session_stats": {},
            "category_affinities": {},
            "recent_events": [],
            "last_feed": [],
            "last_feed_timings": None,
            "message": "No session data. Interact with the feed first."
        }
    return session.to_profile()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
