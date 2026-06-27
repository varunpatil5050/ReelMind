"""
ReelMind Feature Store — Online feature computation and caching layer.

Provides real-time feature vectors for users and videos by combining:
- Redis cache (hot features, TTL-based eviction)
- PostgreSQL (cold features, full history)
- Cross-feature computation (user×video interaction features)

Design:
- Two-level cache: L1 Redis (ms latency) → L2 Postgres (10ms latency)
- Batch feature retrieval for ranking efficiency
- Pre-computed aggregate features refreshed by a background job
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─── Feature Schemas ─────────────────────────────────────────────────────────


@dataclass
class UserFeatures:
    """Real-time user feature vector for ranking models."""

    user_id: str
    # Engagement history (windowed)
    watch_count_1d: int = 0
    watch_count_7d: int = 0
    watch_count_30d: int = 0
    avg_watch_pct_7d: float = 0.0
    like_rate_7d: float = 0.0
    share_rate_7d: float = 0.0
    skip_rate_7d: float = 0.0
    # Session features
    avg_session_length: float = 0.0
    sessions_today: int = 0
    # Category affinity (top 5 categories with scores)
    top_categories: list[tuple[str, float]] = field(default_factory=list)
    # User demographics
    age_group: str = "25-34"
    country: str = "US"
    device: str = "ios"
    engagement_level: float = 0.5
    is_creator: bool = False
    # Recency
    hours_since_last_active: float = 0.0
    account_age_days: int = 0
    # Embedding (if available from Two-Tower)
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "watch_count_1d": self.watch_count_1d,
            "watch_count_7d": self.watch_count_7d,
            "watch_count_30d": self.watch_count_30d,
            "avg_watch_pct_7d": self.avg_watch_pct_7d,
            "like_rate_7d": self.like_rate_7d,
            "share_rate_7d": self.share_rate_7d,
            "skip_rate_7d": self.skip_rate_7d,
            "avg_session_length": self.avg_session_length,
            "sessions_today": self.sessions_today,
            "age_group": self.age_group,
            "country": self.country,
            "device": self.device,
            "engagement_level": self.engagement_level,
            "is_creator": self.is_creator,
            "hours_since_last_active": self.hours_since_last_active,
            "account_age_days": self.account_age_days,
        }

    def to_vector(self) -> np.ndarray:
        """Convert to dense feature vector for model input."""
        numeric = [
            self.watch_count_1d,
            self.watch_count_7d,
            self.watch_count_30d,
            self.avg_watch_pct_7d,
            self.like_rate_7d,
            self.share_rate_7d,
            self.skip_rate_7d,
            self.avg_session_length,
            self.sessions_today,
            self.engagement_level,
            float(self.is_creator),
            self.hours_since_last_active,
            self.account_age_days,
        ]
        return np.array(numeric, dtype=np.float32)


@dataclass
class VideoFeatures:
    """Real-time video feature vector."""

    video_id: str
    creator_id: str = ""
    # Content attributes
    duration_ms: int = 30000
    category: str = "comedy"
    production_quality: float = 0.5
    has_music: bool = True
    has_text_overlay: bool = False
    # Engagement aggregates
    total_views: int = 0
    total_likes: int = 0
    total_shares: int = 0
    total_comments: int = 0
    avg_watch_pct: float = 0.0
    virality_score: float = 0.0
    # Computed rates
    ctr_1d: float = 0.0
    ctr_7d: float = 0.0
    completion_rate_1d: float = 0.0
    completion_rate_7d: float = 0.0
    share_rate_7d: float = 0.0
    # Freshness
    freshness_hours: float = 0.0
    momentum_score: float = 0.0
    # Creator features (denormalized for speed)
    creator_tier: str = "micro"
    creator_follower_count: int = 0
    creator_engagement_rate: float = 0.05
    # Embedding
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "creator_id": self.creator_id,
            "duration_ms": self.duration_ms,
            "category": self.category,
            "production_quality": self.production_quality,
            "has_music": self.has_music,
            "has_text_overlay": self.has_text_overlay,
            "total_views": self.total_views,
            "total_likes": self.total_likes,
            "total_shares": self.total_shares,
            "avg_watch_pct": self.avg_watch_pct,
            "virality_score": self.virality_score,
            "ctr_7d": self.ctr_7d,
            "completion_rate_7d": self.completion_rate_7d,
            "freshness_hours": self.freshness_hours,
            "momentum_score": self.momentum_score,
            "creator_tier": self.creator_tier,
            "creator_follower_count": self.creator_follower_count,
        }

    def to_vector(self) -> np.ndarray:
        """Convert to dense feature vector for model input."""
        numeric = [
            self.duration_ms / 180000.0,  # Normalized
            self.production_quality,
            float(self.has_music),
            float(self.has_text_overlay),
            np.log1p(self.total_views),
            np.log1p(self.total_likes),
            np.log1p(self.total_shares),
            np.log1p(self.total_comments),
            self.avg_watch_pct,
            self.virality_score,
            self.ctr_7d,
            self.completion_rate_7d,
            self.share_rate_7d,
            self.freshness_hours / 720.0,  # Normalized to 30 days
            self.momentum_score,
            np.log1p(self.creator_follower_count),
            self.creator_engagement_rate,
        ]
        return np.array(numeric, dtype=np.float32)


@dataclass
class CrossFeatures:
    """User × Video interaction features for ranking."""

    user_id: str
    video_id: str
    # Category affinity
    category_affinity_score: float = 0.0
    is_preferred_category: bool = False
    # Creator relationship
    is_following_creator: bool = False
    creator_affinity_score: float = 0.0
    # Freshness interaction
    user_recency_x_content_freshness: float = 0.0
    # History features
    user_watched_creator_before: bool = False
    user_category_watch_count: int = 0
    # Diversity
    category_fatigue_score: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.category_affinity_score,
            float(self.is_preferred_category),
            float(self.is_following_creator),
            self.creator_affinity_score,
            self.user_recency_x_content_freshness,
            float(self.user_watched_creator_before),
            self.user_category_watch_count,
            self.category_fatigue_score,
        ], dtype=np.float32)


# ─── Feature Store ────────────────────────────────────────────────────────────


class FeatureStore:
    """
    Two-level feature store with Redis L1 cache and Postgres L2 storage.

    Falls back to synthetic features when infrastructure is unavailable,
    enabling local development and testing without Docker.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        postgres_dsn: str = "postgresql://reelmind:reelmind_dev@localhost:5432/reelmind",
        cache_ttl_seconds: int = 300,
    ):
        self.redis_url = redis_url
        self.postgres_dsn = postgres_dsn
        self.cache_ttl = cache_ttl_seconds
        self._redis = None
        self._pg_pool = None
        self._local_cache: dict[str, Any] = {}
        self._rng = np.random.default_rng(42)

        # Metrics
        self._cache_hits = 0
        self._cache_misses = 0
        self._pg_queries = 0

    async def initialize(self) -> None:
        """Connect to Redis and Postgres. Falls back gracefully."""
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self.redis_url, decode_responses=True, socket_timeout=2
            )
            await self._redis.ping()
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis unavailable, using local cache: {e}")
            self._redis = None

        try:
            import asyncpg
            self._pg_pool = await asyncpg.create_pool(
                self.postgres_dsn, min_size=2, max_size=10, command_timeout=5
            )
            logger.info("PostgreSQL connected")
        except Exception as e:
            logger.warning(f"PostgreSQL unavailable, using synthetic features: {e}")
            self._pg_pool = None

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
        if self._pg_pool:
            await self._pg_pool.close()

    # ─── User Features ───────────────────────────────────────────────────

    async def get_user_features(self, user_id: str) -> UserFeatures:
        """
        Retrieve user features with cache hierarchy:
        L1 Redis → L2 Postgres → Synthetic fallback.
        """
        cache_key = f"uf:{user_id}"

        # L1: Redis cache
        if self._redis:
            try:
                cached = await self._redis.hgetall(cache_key)
                if cached:
                    self._cache_hits += 1
                    return self._parse_user_features(user_id, cached)
            except Exception:
                pass

        # L2: Postgres
        if self._pg_pool:
            try:
                async with self._pg_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT u.age_group, u.country, u.engagement_level, u.is_creator,
                               u.signup_ts,
                               uf.watch_count_1d, uf.watch_count_7d, uf.watch_count_30d,
                               uf.avg_watch_pct_7d, uf.like_rate_7d, uf.share_rate_7d,
                               uf.skip_rate_7d, uf.avg_session_len,
                               uf.top_category_1, uf.top_category_2, uf.top_category_3,
                               uf.last_active_ts
                        FROM users u
                        LEFT JOIN user_features uf ON u.user_id = uf.user_id
                        WHERE u.user_id = $1
                        """,
                        user_id,
                    )
                    if row:
                        self._pg_queries += 1
                        features = self._row_to_user_features(user_id, row)
                        await self._cache_user_features(cache_key, features)
                        return features
            except Exception as e:
                logger.warning(f"Postgres query failed: {e}")

        # Fallback: synthetic features
        self._cache_misses += 1
        return self._synthetic_user_features(user_id)

    async def get_user_features_batch(
        self, user_ids: list[str]
    ) -> dict[str, UserFeatures]:
        """Batch retrieve user features."""
        results = {}
        for uid in user_ids:
            results[uid] = await self.get_user_features(uid)
        return results

    # ─── Video Features ──────────────────────────────────────────────────

    async def get_video_features(self, video_id: str) -> VideoFeatures:
        """Retrieve video features with cache hierarchy."""
        cache_key = f"vf:{video_id}"

        # L1: Redis
        if self._redis:
            try:
                cached = await self._redis.hgetall(cache_key)
                if cached:
                    self._cache_hits += 1
                    return self._parse_video_features(video_id, cached)
            except Exception:
                pass

        # L2: Postgres
        if self._pg_pool:
            try:
                async with self._pg_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT v.creator_id, v.duration_ms, v.category, v.quality_score,
                               v.virality_score, v.total_views, v.total_likes,
                               v.total_shares, v.avg_watch_pct, v.upload_ts,
                               vf.ctr_1d, vf.ctr_7d, vf.completion_rate_1d,
                               vf.completion_rate_7d, vf.share_rate_7d,
                               vf.freshness_hours, vf.momentum_score,
                               c.creator_tier, c.follower_count, c.engagement_rate
                        FROM videos v
                        LEFT JOIN video_features vf ON v.video_id = vf.video_id
                        LEFT JOIN creators c ON v.creator_id = c.creator_id
                        WHERE v.video_id = $1
                        """,
                        video_id,
                    )
                    if row:
                        self._pg_queries += 1
                        features = self._row_to_video_features(video_id, row)
                        await self._cache_video_features(cache_key, features)
                        return features
            except Exception as e:
                logger.warning(f"Postgres query failed: {e}")

        self._cache_misses += 1
        return self._synthetic_video_features(video_id)

    async def get_video_features_batch(
        self, video_ids: list[str]
    ) -> dict[str, VideoFeatures]:
        """Batch retrieve video features."""
        results = {}
        for vid in video_ids:
            results[vid] = await self.get_video_features(vid)
        return results

    # ─── Cross Features ──────────────────────────────────────────────────

    async def compute_cross_features(
        self,
        user_features: UserFeatures,
        video_features: VideoFeatures,
    ) -> CrossFeatures:
        """Compute user×video interaction features."""
        # Category affinity
        user_cats = {cat: score for cat, score in user_features.top_categories}
        cat_affinity = user_cats.get(video_features.category, 0.0)
        is_preferred = video_features.category in user_cats

        # Freshness interaction
        recency_freshness = (
            max(0, 1.0 - user_features.hours_since_last_active / 168.0)
            * max(0, 1.0 - video_features.freshness_hours / 168.0)
        )

        # Category fatigue (diminishing returns from same category)
        cat_fatigue = 1.0 - min(
            user_features.watch_count_7d * cat_affinity / 50.0, 0.8
        )

        return CrossFeatures(
            user_id=user_features.user_id,
            video_id=video_features.video_id,
            category_affinity_score=cat_affinity,
            is_preferred_category=is_preferred,
            user_recency_x_content_freshness=recency_freshness,
            category_fatigue_score=cat_fatigue,
        )

    # ─── Internal Helpers ─────────────────────────────────────────────────

    def _synthetic_user_features(self, user_id: str) -> UserFeatures:
        """Generate realistic synthetic features for local development."""
        idx = int(user_id.split("_")[1]) if "_" in user_id else 0
        rng = np.random.default_rng(idx)

        categories = [
            "comedy", "dance", "education", "food", "gaming",
            "music", "sports", "tech", "fashion", "travel",
        ]
        n_cats = rng.integers(1, 4)
        top_cats = [
            (categories[i], round(float(rng.uniform(0.3, 1.0)), 3))
            for i in rng.choice(len(categories), size=n_cats, replace=False)
        ]

        return UserFeatures(
            user_id=user_id,
            watch_count_1d=int(rng.integers(0, 50)),
            watch_count_7d=int(rng.integers(10, 200)),
            watch_count_30d=int(rng.integers(50, 800)),
            avg_watch_pct_7d=round(float(rng.uniform(0.3, 0.8)), 3),
            like_rate_7d=round(float(rng.uniform(0.02, 0.15)), 3),
            share_rate_7d=round(float(rng.uniform(0.005, 0.03)), 3),
            skip_rate_7d=round(float(rng.uniform(0.1, 0.5)), 3),
            avg_session_length=round(float(rng.uniform(5, 30)), 1),
            sessions_today=int(rng.integers(0, 5)),
            top_categories=top_cats,
            engagement_level=round(float(rng.beta(2, 5)), 3),
            hours_since_last_active=round(float(rng.exponential(12)), 1),
            account_age_days=int(rng.integers(1, 365)),
        )

    def _synthetic_video_features(self, video_id: str) -> VideoFeatures:
        """Generate realistic synthetic video features."""
        idx = int(video_id.split("_")[1]) if "_" in video_id else 0
        rng = np.random.default_rng(idx + 100_000)

        categories = [
            "comedy", "dance", "education", "food", "gaming",
            "music", "sports", "tech", "fashion", "travel",
        ]

        views = int(rng.lognormal(8, 2))
        likes = int(views * rng.uniform(0.02, 0.12))

        return VideoFeatures(
            video_id=video_id,
            creator_id=f"c_{rng.integers(0, 5000)}",
            duration_ms=int(rng.integers(5000, 180000)),
            category=rng.choice(categories),
            production_quality=round(float(rng.uniform(0.2, 0.95)), 3),
            has_music=bool(rng.random() < 0.85),
            has_text_overlay=bool(rng.random() < 0.4),
            total_views=views,
            total_likes=likes,
            total_shares=int(likes * rng.uniform(0.1, 0.4)),
            total_comments=int(likes * rng.uniform(0.05, 0.3)),
            avg_watch_pct=round(float(rng.uniform(0.25, 0.75)), 3),
            virality_score=round(float(rng.beta(2, 20)), 4),
            ctr_7d=round(float(rng.uniform(0.02, 0.15)), 4),
            completion_rate_7d=round(float(rng.uniform(0.3, 0.7)), 4),
            share_rate_7d=round(float(rng.uniform(0.005, 0.03)), 4),
            freshness_hours=round(float(rng.exponential(48)), 1),
            momentum_score=round(float(rng.uniform(0.0, 1.0)), 4),
            creator_tier=rng.choice(
                ["nano", "micro", "mid", "macro", "mega"],
                p=[0.50, 0.30, 0.12, 0.06, 0.02],
            ),
            creator_follower_count=int(rng.lognormal(7, 2)),
            creator_engagement_rate=round(float(rng.beta(2, 40)), 4),
        )

    def _parse_user_features(
        self, user_id: str, data: dict[str, str]
    ) -> UserFeatures:
        """Parse Redis hash into UserFeatures."""
        return UserFeatures(
            user_id=user_id,
            watch_count_1d=int(data.get("watch_count_1d", 0)),
            watch_count_7d=int(data.get("watch_count_7d", 0)),
            watch_count_30d=int(data.get("watch_count_30d", 0)),
            avg_watch_pct_7d=float(data.get("avg_watch_pct_7d", 0)),
            like_rate_7d=float(data.get("like_rate_7d", 0)),
            share_rate_7d=float(data.get("share_rate_7d", 0)),
            skip_rate_7d=float(data.get("skip_rate_7d", 0)),
            avg_session_length=float(data.get("avg_session_length", 0)),
            engagement_level=float(data.get("engagement_level", 0.5)),
        )

    def _parse_video_features(
        self, video_id: str, data: dict[str, str]
    ) -> VideoFeatures:
        """Parse Redis hash into VideoFeatures."""
        return VideoFeatures(
            video_id=video_id,
            creator_id=data.get("creator_id", ""),
            duration_ms=int(data.get("duration_ms", 30000)),
            category=data.get("category", "comedy"),
            production_quality=float(data.get("production_quality", 0.5)),
            total_views=int(data.get("total_views", 0)),
            total_likes=int(data.get("total_likes", 0)),
            avg_watch_pct=float(data.get("avg_watch_pct", 0)),
            virality_score=float(data.get("virality_score", 0)),
            ctr_7d=float(data.get("ctr_7d", 0)),
            completion_rate_7d=float(data.get("completion_rate_7d", 0)),
            freshness_hours=float(data.get("freshness_hours", 0)),
            momentum_score=float(data.get("momentum_score", 0)),
        )

    def _row_to_user_features(self, user_id: str, row: Any) -> UserFeatures:
        """Convert Postgres row to UserFeatures."""
        now_ms = int(time.time() * 1000)
        signup_ts = row.get("signup_ts", now_ms)
        last_active = row.get("last_active_ts") or now_ms

        top_cats = []
        for i in range(1, 4):
            cat = row.get(f"top_category_{i}")
            if cat:
                top_cats.append((cat, 1.0 - i * 0.2))

        return UserFeatures(
            user_id=user_id,
            watch_count_1d=row.get("watch_count_1d", 0) or 0,
            watch_count_7d=row.get("watch_count_7d", 0) or 0,
            watch_count_30d=row.get("watch_count_30d", 0) or 0,
            avg_watch_pct_7d=row.get("avg_watch_pct_7d", 0) or 0,
            like_rate_7d=row.get("like_rate_7d", 0) or 0,
            share_rate_7d=row.get("share_rate_7d", 0) or 0,
            skip_rate_7d=row.get("skip_rate_7d", 0) or 0,
            avg_session_length=row.get("avg_session_len", 0) or 0,
            top_categories=top_cats,
            age_group=row.get("age_group", "25-34"),
            country=row.get("country", "US"),
            engagement_level=row.get("engagement_level", 0.5) or 0.5,
            is_creator=row.get("is_creator", False) or False,
            hours_since_last_active=(now_ms - last_active) / 3_600_000,
            account_age_days=(now_ms - signup_ts) // 86_400_000,
        )

    def _row_to_video_features(self, video_id: str, row: Any) -> VideoFeatures:
        """Convert Postgres row to VideoFeatures."""
        return VideoFeatures(
            video_id=video_id,
            creator_id=row.get("creator_id", ""),
            duration_ms=row.get("duration_ms", 30000) or 30000,
            category=row.get("category", "comedy"),
            production_quality=row.get("quality_score", 0.5) or 0.5,
            total_views=row.get("total_views", 0) or 0,
            total_likes=row.get("total_likes", 0) or 0,
            total_shares=row.get("total_shares", 0) or 0,
            avg_watch_pct=row.get("avg_watch_pct", 0) or 0,
            virality_score=row.get("virality_score", 0) or 0,
            ctr_1d=row.get("ctr_1d", 0) or 0,
            ctr_7d=row.get("ctr_7d", 0) or 0,
            completion_rate_1d=row.get("completion_rate_1d", 0) or 0,
            completion_rate_7d=row.get("completion_rate_7d", 0) or 0,
            share_rate_7d=row.get("share_rate_7d", 0) or 0,
            freshness_hours=row.get("freshness_hours", 0) or 0,
            momentum_score=row.get("momentum_score", 0) or 0,
            creator_tier=row.get("creator_tier", "micro") or "micro",
            creator_follower_count=row.get("follower_count", 0) or 0,
            creator_engagement_rate=row.get("engagement_rate", 0.05) or 0.05,
        )

    async def _cache_user_features(
        self, key: str, features: UserFeatures
    ) -> None:
        """Write user features to Redis cache."""
        if not self._redis:
            return
        try:
            data = {
                "watch_count_1d": str(features.watch_count_1d),
                "watch_count_7d": str(features.watch_count_7d),
                "watch_count_30d": str(features.watch_count_30d),
                "avg_watch_pct_7d": str(features.avg_watch_pct_7d),
                "like_rate_7d": str(features.like_rate_7d),
                "share_rate_7d": str(features.share_rate_7d),
                "skip_rate_7d": str(features.skip_rate_7d),
                "avg_session_length": str(features.avg_session_length),
                "engagement_level": str(features.engagement_level),
            }
            await self._redis.hset(key, mapping=data)
            await self._redis.expire(key, self.cache_ttl)
        except Exception as e:
            logger.warning(f"Redis cache write failed: {e}")

    async def _cache_video_features(
        self, key: str, features: VideoFeatures
    ) -> None:
        """Write video features to Redis cache."""
        if not self._redis:
            return
        try:
            data = {
                "creator_id": features.creator_id,
                "duration_ms": str(features.duration_ms),
                "category": features.category,
                "production_quality": str(features.production_quality),
                "total_views": str(features.total_views),
                "total_likes": str(features.total_likes),
                "avg_watch_pct": str(features.avg_watch_pct),
                "virality_score": str(features.virality_score),
                "ctr_7d": str(features.ctr_7d),
                "completion_rate_7d": str(features.completion_rate_7d),
                "freshness_hours": str(features.freshness_hours),
                "momentum_score": str(features.momentum_score),
            }
            await self._redis.hset(key, mapping=data)
            await self._redis.expire(key, self.cache_ttl)
        except Exception as e:
            logger.warning(f"Redis cache write failed: {e}")

    @property
    def cache_stats(self) -> dict[str, int]:
        total = self._cache_hits + self._cache_misses
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": round(self._cache_hits / max(total, 1), 4),
            "pg_queries": self._pg_queries,
        }
