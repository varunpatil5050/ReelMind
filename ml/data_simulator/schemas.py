"""
ReelMind Event Schemas — Pydantic models for all system events.

These schemas enforce strict typing across the entire pipeline:
Kafka producers → Feature engine → Training data → Evaluation.

Design decisions:
- orjson for high-throughput serialization (~10x faster than stdlib json)
- Frozen models where possible to prevent mutation in streaming contexts
- Explicit field validators to catch data quality issues at ingestion time
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Enums ────────────────────────────────────────────────────────────────────


class EventType(str, enum.Enum):
    WATCH = "watch"
    LIKE = "like"
    SHARE = "share"
    COMMENT = "comment"
    SKIP = "skip"
    FOLLOW_CREATOR = "follow_creator"
    SAVE = "save"


class DeviceType(str, enum.Enum):
    IOS = "ios"
    ANDROID = "android"
    WEB = "web"


class FeedSource(str, enum.Enum):
    FOR_YOU = "for_you"
    FOLLOWING = "following"
    SEARCH = "search"
    TRENDING = "trending"
    CREATOR_PAGE = "creator_page"


class ContentCategory(str, enum.Enum):
    COMEDY = "comedy"
    DANCE = "dance"
    EDUCATION = "education"
    FOOD = "food"
    GAMING = "gaming"
    MUSIC = "music"
    SPORTS = "sports"
    TECH = "tech"
    FASHION = "fashion"
    TRAVEL = "travel"
    FITNESS = "fitness"
    PETS = "pets"
    NEWS = "news"
    DIY = "diy"
    BEAUTY = "beauty"


class UserAgeGroup(str, enum.Enum):
    TEEN = "13-17"
    YOUNG_ADULT = "18-24"
    ADULT = "25-34"
    MID_ADULT = "35-44"
    SENIOR = "45+"


# ─── Context Models ──────────────────────────────────────────────────────────


class EventContext(BaseModel):
    """Contextual signals captured at event time."""

    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    feed_position: int = Field(ge=0, description="0-indexed position in feed")
    source: FeedSource = FeedSource.FOR_YOU
    is_wifi: bool = True
    app_version: str = "1.0.0"

    model_config = {"frozen": True}


class GeoInfo(BaseModel):
    """Geographic context for the event."""

    country: str = Field(max_length=2, description="ISO 3166-1 alpha-2")
    region: str = Field(max_length=10)
    city: Optional[str] = None

    model_config = {"frozen": True}


# ─── Core Event Schema ───────────────────────────────────────────────────────


class UserEvent(BaseModel):
    """
    Primary event schema for all user interactions.
    
    This is the canonical schema consumed by:
    - Kafka topic: user.events.raw
    - Feature engine for online feature computation
    - Training pipeline for label extraction
    - RL optimizer for reward computation
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(pattern=r"^u_\d+$")
    video_id: str = Field(pattern=r"^v_\d+$")
    creator_id: str = Field(pattern=r"^c_\d+$")
    event_type: EventType
    timestamp_ms: int = Field(gt=0)
    session_id: str = Field(pattern=r"^sess_[a-z0-9]+$")

    # Engagement signals
    watch_duration_ms: int = Field(ge=0, default=0)
    video_duration_ms: int = Field(gt=0)
    watch_percentage: float = Field(ge=0.0, le=2.0, default=0.0)
    is_replay: bool = False
    replay_count: int = Field(ge=0, default=0)

    # Device & geo
    device: DeviceType = DeviceType.IOS
    geo: GeoInfo = GeoInfo(country="US", region="US-CA")

    # Context
    context: EventContext

    @field_validator("watch_percentage", mode="before")
    @classmethod
    def compute_watch_pct(cls, v: float, info) -> float:
        """Allow >1.0 for replays (up to 2.0)."""
        if v < 0:
            return 0.0
        return min(v, 2.0)

    @model_validator(mode="after")
    def validate_engagement_consistency(self) -> "UserEvent":
        """Ensure watch signals are consistent with event type."""
        if self.event_type == EventType.SKIP and self.watch_percentage > 0.25:
            object.__setattr__(self, "watch_percentage", min(self.watch_percentage, 0.25))
        return self

    def to_kafka_key(self) -> bytes:
        return self.user_id.encode("utf-8")

    def to_kafka_value(self) -> bytes:
        import orjson
        return orjson.dumps(self.model_dump(mode="json"))


# ─── Entity Schemas ───────────────────────────────────────────────────────────


class UserProfile(BaseModel):
    """Static + slowly-changing user attributes."""

    user_id: str = Field(pattern=r"^u_\d+$")
    age_group: UserAgeGroup
    country: str = Field(max_length=2)
    signup_timestamp_ms: int = Field(gt=0)
    preferred_categories: list[ContentCategory] = Field(max_length=5)
    follower_count: int = Field(ge=0, default=0)
    following_count: int = Field(ge=0, default=0)
    total_watch_time_hours: float = Field(ge=0.0, default=0.0)
    avg_session_duration_min: float = Field(ge=0.0, default=0.0)
    engagement_level: float = Field(
        ge=0.0, le=1.0, default=0.5,
        description="0=passive lurker, 1=power user"
    )
    is_creator: bool = False
    embedding: Optional[list[float]] = Field(
        default=None, description="User embedding vector (128d)"
    )


class VideoMetadata(BaseModel):
    """Content metadata for videos in the catalog."""

    video_id: str = Field(pattern=r"^v_\d+$")
    creator_id: str = Field(pattern=r"^c_\d+$")
    duration_ms: int = Field(gt=0, le=180000, description="Max 3 min")
    category: ContentCategory
    tags: list[str] = Field(max_length=10)
    upload_timestamp_ms: int = Field(gt=0)
    language: str = Field(default="en", max_length=5)

    # Content quality signals (simulated)
    production_quality: float = Field(ge=0.0, le=1.0)
    audio_quality: float = Field(ge=0.0, le=1.0)
    has_text_overlay: bool = False
    has_music: bool = True

    # Aggregate engagement (updated periodically)
    total_views: int = Field(ge=0, default=0)
    total_likes: int = Field(ge=0, default=0)
    total_shares: int = Field(ge=0, default=0)
    total_comments: int = Field(ge=0, default=0)
    avg_watch_percentage: float = Field(ge=0.0, le=2.0, default=0.0)
    virality_score: float = Field(ge=0.0, le=1.0, default=0.0)

    embedding: Optional[list[float]] = Field(
        default=None, description="Item embedding vector (128d)"
    )


class CreatorProfile(BaseModel):
    """Creator entity with content production patterns."""

    creator_id: str = Field(pattern=r"^c_\d+$")
    user_id: str = Field(pattern=r"^u_\d+$")
    niche_categories: list[ContentCategory] = Field(min_length=1, max_length=3)
    follower_count: int = Field(ge=0, default=0)
    total_videos: int = Field(ge=0, default=0)
    avg_video_quality: float = Field(ge=0.0, le=1.0, default=0.5)
    upload_frequency_per_week: float = Field(ge=0.0, default=1.0)
    engagement_rate: float = Field(ge=0.0, le=1.0, default=0.05)
    creator_tier: str = Field(default="micro", pattern=r"^(nano|micro|mid|macro|mega)$")


# ─── Impression / Recommendation Schemas ──────────────────────────────────────


class ImpressionLog(BaseModel):
    """Logged when recommendations are served — critical for RL and evaluation."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(pattern=r"^u_\d+$")
    timestamp_ms: int = Field(gt=0)
    session_id: str

    # What was shown
    served_video_ids: list[str] = Field(min_length=1, max_length=30)
    served_scores: list[float] = Field(min_length=1, max_length=30)

    # Pipeline metadata
    retrieval_candidates_count: int = Field(ge=0)
    retrieval_latency_ms: float = Field(ge=0)
    ranking_latency_ms: float = Field(ge=0)
    reranking_latency_ms: float = Field(ge=0)
    total_latency_ms: float = Field(ge=0)

    # RL context
    policy_version: str = "v0"
    exploration_flag: bool = False

    model_config = {"frozen": True}


class RecommendationRequest(BaseModel):
    """API request schema for feed generation."""

    user_id: str = Field(pattern=r"^u_\d+$")
    num_results: int = Field(ge=1, le=30, default=15)
    feed_source: FeedSource = FeedSource.FOR_YOU
    session_id: Optional[str] = None
    exclude_video_ids: list[str] = Field(default_factory=list, max_length=100)
    device: DeviceType = DeviceType.IOS


class RecommendationResponse(BaseModel):
    """API response schema for feed generation."""

    request_id: str
    user_id: str
    videos: list[ScoredVideo]
    total_latency_ms: float
    retrieval_latency_ms: float
    ranking_latency_ms: float
    model_version: str
    policy_version: str


class ScoredVideo(BaseModel):
    """A video with its ranking score and explanation."""

    video_id: str
    score: float
    rank: int
    retrieval_source: str = "two_tower"
    predicted_watch_pct: float = Field(ge=0.0, le=2.0)
    predicted_ctr: float = Field(ge=0.0, le=1.0)
    diversity_penalty: float = Field(ge=0.0, le=1.0, default=0.0)


# ─── Model Signal Schemas ────────────────────────────────────────────────────


class ModelSignal(BaseModel):
    """Emitted by ops agents when drift or degradation is detected."""

    signal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str
    signal_type: str = Field(pattern=r"^(drift|degradation|anomaly|retrain_needed)$")
    severity: str = Field(pattern=r"^(low|medium|high|critical)$")
    timestamp_ms: int = Field(gt=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    message: str = ""
    auto_action_taken: Optional[str] = None
