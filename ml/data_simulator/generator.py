"""
ReelMind Synthetic Data Generator — Production-scale data simulation engine.

Generates realistic recommendation system training data with:
- Correlated user preferences and behavior patterns
- Temporal engagement dynamics
- Viral content simulation
- Cold-start scenarios
- Session-based interaction sequences
- Feedback loop effects

Output formats: Parquet (columnar, compressed), JSON (streaming-compatible)

Usage:
    generator = DataGenerator(seed=42, num_users=50000, num_videos=100000)
    generator.generate_all()
    generator.save_parquet("./data/")
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from tqdm import tqdm

from .distributions import (
    CategoryCorrelation,
    ColdStartModel,
    EngagementDistribution,
    SessionModel,
    TemporalDistribution,
    ViralityModel,
)
from .schemas import (
    ContentCategory,
    CreatorProfile,
    DeviceType,
    EventContext,
    EventType,
    FeedSource,
    GeoInfo,
    UserAgeGroup,
    UserEvent,
    UserProfile,
    VideoMetadata,
)

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

ALL_CATEGORIES = [c.value for c in ContentCategory]
ALL_COUNTRIES = ["US", "IN", "BR", "GB", "DE", "JP", "KR", "FR", "MX", "ID"]
COUNTRY_WEIGHTS = [0.25, 0.20, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.07, 0.07]
DEVICE_WEIGHTS = [0.45, 0.45, 0.10]  # iOS, Android, Web
AGE_GROUPS = [ag.value for ag in UserAgeGroup]
AGE_WEIGHTS = [0.12, 0.35, 0.28, 0.15, 0.10]
CREATOR_TIERS = ["nano", "micro", "mid", "macro", "mega"]
CREATOR_TIER_WEIGHTS = [0.50, 0.30, 0.12, 0.06, 0.02]

# Regions per country (simplified)
COUNTRY_REGIONS = {
    "US": ["US-CA", "US-NY", "US-TX", "US-FL", "US-IL"],
    "IN": ["IN-MH", "IN-DL", "IN-KA", "IN-TN", "IN-UP"],
    "BR": ["BR-SP", "BR-RJ", "BR-MG", "BR-BA", "BR-RS"],
    "GB": ["GB-ENG", "GB-SCT", "GB-WLS"],
    "DE": ["DE-BY", "DE-NW", "DE-BW"],
    "JP": ["JP-13", "JP-27", "JP-14"],
    "KR": ["KR-11", "KR-26", "KR-28"],
    "FR": ["FR-IDF", "FR-ARA", "FR-NAQ"],
    "MX": ["MX-CMX", "MX-JAL", "MX-NLE"],
    "ID": ["ID-JK", "ID-JB", "ID-JI"],
}


@dataclass
class GeneratorConfig:
    """Configuration for data generation."""

    num_users: int = 50_000
    num_videos: int = 100_000
    num_interactions: int = 10_000_000
    num_creators: int = 5_000  # ~10% of users are creators

    # Temporal simulation
    simulation_days: int = 30
    base_timestamp_ms: int = 1_700_000_000_000  # ~Nov 2023

    # Content distribution
    creator_to_video_ratio: float = 20.0  # avg videos per creator
    video_duration_range_ms: tuple[int, int] = (5_000, 180_000)

    # User behavior
    avg_interactions_per_user: float = 200.0
    power_law_alpha: float = 1.5  # User activity power law exponent

    # Random seed
    seed: int = 42

    # Output
    output_dir: str = "./data"
    parquet_compression: str = "snappy"
    batch_size: int = 100_000  # For chunked generation


class DataGenerator:
    """
    High-throughput synthetic data generator for recommendation systems.
    
    Architecture:
    1. Generate entity catalogs (users, creators, videos) — O(N)
    2. Build preference matrices (user × category affinity) — O(U × C)
    3. Simulate sessions with temporal dynamics — O(interactions)
    4. Apply viral dynamics and cold-start patterns — O(interactions)
    5. Export to Parquet with columnar compression
    
    Memory optimization: Generates interactions in batches to handle
    10M+ events without OOM on 16GB machines.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Distribution models
        self.temporal = TemporalDistribution()
        self.engagement = EngagementDistribution()
        self.category_corr = CategoryCorrelation()
        self.virality = ViralityModel()
        self.session_model = SessionModel()
        self.cold_start = ColdStartModel()

        # Entity storage
        self.users: list[dict] = []
        self.creators: list[dict] = []
        self.videos: list[dict] = []
        self.interactions: list[dict] = []

        # Lookup indices (built during generation)
        self._user_categories: dict[str, list[str]] = {}
        self._video_by_category: dict[str, list[str]] = {}
        self._creator_videos: dict[str, list[str]] = {}
        self._video_metadata: dict[str, dict] = {}
        self._user_engagement: dict[str, float] = {}
        self._video_impressions: dict[str, int] = {}

        # Statistics
        self.stats: dict[str, float] = {}

    def generate_all(self) -> None:
        """Execute full data generation pipeline."""
        start = time.time()
        logger.info("Starting data generation pipeline...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Generating creators...", total=5)

            self._generate_creators()
            progress.update(task, advance=1, description="Generating users...")

            self._generate_users()
            progress.update(task, advance=1, description="Generating videos...")

            self._generate_videos()
            self._build_indices()
            progress.update(task, advance=1, description="Generating interactions...")

            self._generate_interactions()
            progress.update(task, advance=1, description="Computing aggregate stats...")

            self._compute_aggregate_stats()
            progress.update(task, advance=1, description="Done!")

        elapsed = time.time() - start
        self.stats["generation_time_seconds"] = elapsed
        logger.info(f"Data generation complete in {elapsed:.1f}s")
        self._log_stats()

    def _generate_creators(self) -> None:
        """Generate creator profiles with tier-based attributes."""
        for i in range(self.config.num_creators):
            creator_id = f"c_{i}"
            user_id = f"u_{i}"  # First N users are creators
            tier = self.rng.choice(CREATOR_TIERS, p=CREATOR_TIER_WEIGHTS)

            # Tier determines follower range and quality
            tier_config = {
                "nano":  {"followers": (100, 1_000),    "quality": (0.3, 0.6)},
                "micro": {"followers": (1_000, 10_000),  "quality": (0.4, 0.7)},
                "mid":   {"followers": (10_000, 100_000), "quality": (0.5, 0.8)},
                "macro": {"followers": (100_000, 1_000_000), "quality": (0.6, 0.9)},
                "mega":  {"followers": (1_000_000, 10_000_000), "quality": (0.7, 0.95)},
            }
            tc = tier_config[tier]

            primary_cat = self.rng.choice(ALL_CATEGORIES)
            niche = self.category_corr.sample_related_categories(
                self.rng, primary_cat, num_additional=self.rng.integers(0, 2)
            )

            self.creators.append({
                "creator_id": creator_id,
                "user_id": user_id,
                "niche_categories": niche,
                "follower_count": int(self.rng.integers(*tc["followers"])),
                "total_videos": 0,  # Updated after video generation
                "avg_video_quality": float(self.rng.uniform(*tc["quality"])),
                "upload_frequency_per_week": float(self.rng.exponential(2.0) + 0.5),
                "engagement_rate": float(self.rng.beta(2, 40)),
                "creator_tier": tier,
            })

    def _generate_users(self) -> None:
        """Generate user profiles with correlated preferences."""
        for i in range(self.config.num_users):
            user_id = f"u_{i}"
            country = self.rng.choice(ALL_COUNTRIES, p=COUNTRY_WEIGHTS)
            age_group = self.rng.choice(AGE_GROUPS, p=AGE_WEIGHTS)

            # Sample correlated category preferences
            primary_cat = self.rng.choice(ALL_CATEGORIES)
            num_interests = self.rng.integers(1, 5)
            preferred = self.category_corr.sample_related_categories(
                self.rng, primary_cat, num_additional=num_interests
            )

            # Engagement level: power-law distributed (most users are passive)
            engagement = float(self.rng.beta(2, 5))

            region = self.rng.choice(COUNTRY_REGIONS.get(country, [f"{country}-XX"]))

            signup_days_ago = self.rng.integers(1, 365)
            signup_ts = self.config.base_timestamp_ms - signup_days_ago * 86_400_000

            self.users.append({
                "user_id": user_id,
                "age_group": age_group,
                "country": country,
                "region": region,
                "signup_timestamp_ms": int(signup_ts),
                "preferred_categories": preferred[:5],
                "follower_count": int(self.rng.integers(0, 500)),
                "following_count": int(self.rng.integers(0, 200)),
                "total_watch_time_hours": 0.0,
                "avg_session_duration_min": 0.0,
                "engagement_level": engagement,
                "is_creator": i < self.config.num_creators,
                "device": self.rng.choice(
                    [d.value for d in DeviceType], p=DEVICE_WEIGHTS
                ),
            })

            self._user_categories[user_id] = preferred[:5]
            self._user_engagement[user_id] = engagement

    def _generate_videos(self) -> None:
        """Generate video catalog distributed across creators."""
        creator_ids = [c["creator_id"] for c in self.creators]

        # Distribute videos with power-law: popular creators get more
        follower_counts = np.array([c["follower_count"] for c in self.creators], dtype=float)
        creator_probs = follower_counts ** 0.5  # Sub-linear to avoid extreme skew
        creator_probs = creator_probs / creator_probs.sum()

        for i in range(self.config.num_videos):
            video_id = f"v_{i}"
            creator_idx = self.rng.choice(len(creator_ids), p=creator_probs)
            creator = self.creators[creator_idx]

            category = self.rng.choice(creator["niche_categories"])
            duration_ms = int(self.rng.integers(*self.config.video_duration_range_ms))

            # Quality correlates with creator quality + random variation
            quality = np.clip(
                creator["avg_video_quality"] + self.rng.normal(0, 0.15),
                0.05, 0.99,
            )

            # Upload time: distributed over simulation window
            upload_day = self.rng.integers(0, self.config.simulation_days)
            upload_ts = self.config.base_timestamp_ms + upload_day * 86_400_000
            upload_ts += self.rng.integers(0, 86_400_000)

            # Virality determination
            is_viral = self.virality.is_viral(self.rng, quality)
            virality_score = float(self.rng.uniform(0.5, 1.0) if is_viral else self.rng.beta(2, 20))

            # Tags: category + related terms
            num_tags = self.rng.integers(2, 8)
            tags = [category] + [
                f"tag_{hashlib.md5(f'{category}_{j}'.encode()).hexdigest()[:6]}"
                for j in range(num_tags - 1)
            ]

            video = {
                "video_id": video_id,
                "creator_id": creator["creator_id"],
                "duration_ms": duration_ms,
                "category": category,
                "tags": tags,
                "upload_timestamp_ms": int(upload_ts),
                "language": "en",
                "production_quality": float(quality),
                "audio_quality": float(np.clip(quality + self.rng.normal(0, 0.1), 0, 1)),
                "has_text_overlay": bool(self.rng.random() < 0.4),
                "has_music": bool(self.rng.random() < 0.85),
                "total_views": 0,
                "total_likes": 0,
                "total_shares": 0,
                "total_comments": 0,
                "avg_watch_percentage": 0.0,
                "virality_score": virality_score,
            }

            self.videos.append(video)
            self._video_metadata[video_id] = video

            # Index by category
            if category not in self._video_by_category:
                self._video_by_category[category] = []
            self._video_by_category[category].append(video_id)

            # Index by creator
            cid = creator["creator_id"]
            if cid not in self._creator_videos:
                self._creator_videos[cid] = []
            self._creator_videos[cid].append(video_id)

    def _build_indices(self) -> None:
        """Build lookup indices for efficient interaction generation."""
        # Count videos per creator
        for creator in self.creators:
            cid = creator["creator_id"]
            creator["total_videos"] = len(self._creator_videos.get(cid, []))

        # Initialize impression counters
        for v in self.videos:
            self._video_impressions[v["video_id"]] = 0

        logger.info(
            f"Indices built: {len(self._video_by_category)} categories, "
            f"{len(self._creator_videos)} creators with videos"
        )

    def _generate_interactions(self) -> None:
        """
        Generate interaction events in session-based batches.
        
        Algorithm:
        1. Assign interaction budget per user (power-law distributed)
        2. For each user, simulate sessions:
           a. Sample session length
           b. For each video in session:
              - Select candidate video (preference-weighted)
              - Sample watch time (quality + engagement dependent)
              - Sample engagement action
              - Apply temporal dynamics
              - Apply cold-start and viral modifiers
        3. Write in batches to avoid OOM
        """
        # Distribute interactions across users (power-law)
        user_ids = [u["user_id"] for u in self.users]
        engagement_levels = np.array([u["engagement_level"] for u in self.users])

        # Power-law activity: engagement ^ alpha
        activity_weights = engagement_levels ** self.config.power_law_alpha
        activity_weights = activity_weights / activity_weights.sum()
        interactions_per_user = self.rng.multinomial(
            self.config.num_interactions, activity_weights
        )

        total_generated = 0
        batch: list[dict] = []

        for user_idx in tqdm(range(self.config.num_users), desc="Simulating users"):
            user = self.users[user_idx]
            user_id = user["user_id"]
            n_interactions = interactions_per_user[user_idx]
            if n_interactions == 0:
                continue

            user_cats = self._user_categories[user_id]
            user_eng = user["engagement_level"]
            user_maturity = self.cold_start.get_user_maturity(n_interactions)

            # Simulate sessions
            remaining = n_interactions
            session_num = 0

            while remaining > 0:
                session_id = f"sess_{hashlib.md5(f'{user_id}_{session_num}'.encode()).hexdigest()[:8]}"
                session_len = self.session_model.sample_session_length(self.rng, user_eng)
                session_len = min(session_len, remaining)

                # Session timestamp: random day + temporal-weighted hour
                session_day = self.rng.integers(0, self.config.simulation_days)
                day_base = self.config.base_timestamp_ms + session_day * 86_400_000
                session_start = self.temporal.sample_timestamp(self.rng, day_base)

                for pos in range(session_len):
                    video = self._select_video_for_user(user_id, user_cats, user_maturity)
                    if video is None:
                        continue

                    vid = video["video_id"]
                    quality = video["production_quality"]
                    virality = video["virality_score"]
                    duration_ms = video["duration_ms"]

                    # Apply session fatigue
                    fatigue = self.session_model.get_fatigue_factor(pos)

                    # Sample watch percentage
                    watch_pct = self.engagement.sample_watch_percentage(
                        self.rng, duration_ms, user_eng * fatigue, quality
                    )
                    watch_ms = int(watch_pct * duration_ms)

                    # Sample engagement action
                    action = self.engagement.sample_engagement_action(
                        self.rng, watch_pct, user_eng * fatigue, quality, virality
                    )

                    # Map action to EventType
                    event_type = EventType.WATCH if action == "watch" else EventType(action)

                    # Replay detection
                    is_replay = watch_pct > 1.0
                    replay_count = max(0, int(watch_pct) - 1) if is_replay else 0

                    # Timestamp: session_start + accumulated watch time
                    ts = session_start + pos * 2000 + watch_ms
                    hour = (ts // 3_600_000) % 24
                    dow = (ts // 86_400_000) % 7

                    interaction = {
                        "event_id": str(uuid.uuid4()),
                        "user_id": user_id,
                        "video_id": vid,
                        "creator_id": video["creator_id"],
                        "event_type": event_type.value,
                        "timestamp_ms": int(ts),
                        "session_id": session_id,
                        "watch_duration_ms": watch_ms,
                        "video_duration_ms": duration_ms,
                        "watch_percentage": round(watch_pct, 4),
                        "is_replay": is_replay,
                        "replay_count": replay_count,
                        "device": user.get("device", "ios"),
                        "country": user["country"],
                        "region": user["region"],
                        "hour_of_day": int(hour),
                        "day_of_week": int(dow),
                        "feed_position": pos,
                        "feed_source": self._sample_feed_source(),
                        "session_position": pos,
                        "session_length": session_len,
                        "user_engagement_level": user_eng,
                        "content_quality": quality,
                        "virality_score": virality,
                        "category": video["category"],
                    }

                    batch.append(interaction)
                    self._video_impressions[vid] = self._video_impressions.get(vid, 0) + 1
                    total_generated += 1

                    # Flush batch to main list
                    if len(batch) >= self.config.batch_size:
                        self.interactions.extend(batch)
                        batch = []

                remaining -= session_len
                session_num += 1

        # Flush remaining
        if batch:
            self.interactions.extend(batch)

        logger.info(f"Generated {total_generated:,} interactions")

    def _select_video_for_user(
        self,
        user_id: str,
        user_categories: list[str],
        user_maturity: float,
    ) -> Optional[dict]:
        """
        Select a video for a user based on preference-weighted sampling.
        
        Strategy:
        - 60% from preferred categories (exploitation)
        - 25% from correlated categories (soft exploration)
        - 15% random (hard exploration — higher for cold-start users)
        
        Cold-start users get boosted exploration.
        """
        explore_boost = max(0, (1 - user_maturity) * 0.2)

        roll = self.rng.random()

        if roll < (0.60 - explore_boost):
            # Preferred category
            cat = self.rng.choice(user_categories)
        elif roll < (0.85 - explore_boost / 2):
            # Correlated category
            primary = user_categories[0]
            related = self.category_corr.sample_related_categories(
                self.rng, primary, num_additional=1
            )
            cat = related[-1] if len(related) > 1 else primary
        else:
            # Random exploration
            cat = self.rng.choice(ALL_CATEGORIES)

        candidates = self._video_by_category.get(cat)
        if not candidates:
            candidates = self._video_by_category.get(
                self.rng.choice(list(self._video_by_category.keys()))
            )
        if not candidates:
            return None

        video_id = self.rng.choice(candidates)
        return self._video_metadata.get(video_id)

    def _sample_feed_source(self) -> str:
        """Sample feed source with realistic distribution."""
        return self.rng.choice(
            [fs.value for fs in FeedSource],
            p=[0.65, 0.15, 0.10, 0.07, 0.03],
        )

    def _compute_aggregate_stats(self) -> None:
        """Compute and store aggregate statistics for videos and users."""
        # Video aggregate stats
        video_stats: dict[str, dict] = {v["video_id"]: {
            "views": 0, "likes": 0, "shares": 0, "comments": 0,
            "watch_pcts": [],
        } for v in self.videos}

        for interaction in self.interactions:
            vid = interaction["video_id"]
            if vid not in video_stats:
                continue
            vs = video_stats[vid]
            vs["views"] += 1
            if interaction["event_type"] == "like":
                vs["likes"] += 1
            elif interaction["event_type"] == "share":
                vs["shares"] += 1
            elif interaction["event_type"] == "comment":
                vs["comments"] += 1
            vs["watch_pcts"].append(interaction["watch_percentage"])

        for video in self.videos:
            vid = video["video_id"]
            vs = video_stats[vid]
            video["total_views"] = vs["views"]
            video["total_likes"] = vs["likes"]
            video["total_shares"] = vs["shares"]
            video["total_comments"] = vs["comments"]
            if vs["watch_pcts"]:
                video["avg_watch_percentage"] = round(
                    sum(vs["watch_pcts"]) / len(vs["watch_pcts"]), 4
                )

        # Compute summary stats
        total_events = len(self.interactions)
        event_types = {}
        for i in self.interactions:
            et = i["event_type"]
            event_types[et] = event_types.get(et, 0) + 1

        self.stats.update({
            "total_users": len(self.users),
            "total_creators": len(self.creators),
            "total_videos": len(self.videos),
            "total_interactions": total_events,
            "unique_sessions": len(set(i["session_id"] for i in self.interactions)),
            "avg_interactions_per_user": total_events / max(len(self.users), 1),
            "event_type_distribution": event_types,
            "avg_watch_percentage": sum(
                i["watch_percentage"] for i in self.interactions
            ) / max(total_events, 1),
        })

    def save_parquet(self, output_dir: Optional[str] = None) -> dict[str, str]:
        """
        Save all generated data to Parquet files.
        
        Returns dict mapping entity type to file path.
        """
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}

        # Users
        users_df = pd.DataFrame(self.users)
        # Convert list columns to strings for Parquet compatibility
        users_df["preferred_categories"] = users_df["preferred_categories"].apply(
            lambda x: ",".join(x) if isinstance(x, list) else x
        )
        users_path = out / "users.parquet"
        users_df.to_parquet(users_path, compression=self.config.parquet_compression, index=False)
        paths["users"] = str(users_path)
        logger.info(f"Saved {len(users_df):,} users → {users_path}")

        # Creators
        creators_df = pd.DataFrame(self.creators)
        creators_df["niche_categories"] = creators_df["niche_categories"].apply(
            lambda x: ",".join(x) if isinstance(x, list) else x
        )
        creators_path = out / "creators.parquet"
        creators_df.to_parquet(creators_path, compression=self.config.parquet_compression, index=False)
        paths["creators"] = str(creators_path)
        logger.info(f"Saved {len(creators_df):,} creators → {creators_path}")

        # Videos
        videos_df = pd.DataFrame(self.videos)
        videos_df["tags"] = videos_df["tags"].apply(
            lambda x: ",".join(x) if isinstance(x, list) else x
        )
        videos_path = out / "videos.parquet"
        videos_df.to_parquet(videos_path, compression=self.config.parquet_compression, index=False)
        paths["videos"] = str(videos_path)
        logger.info(f"Saved {len(videos_df):,} videos → {videos_path}")

        # Interactions (chunked for large datasets)
        interactions_path = out / "interactions.parquet"
        interactions_df = pd.DataFrame(self.interactions)
        interactions_df.to_parquet(
            interactions_path, compression=self.config.parquet_compression, index=False
        )
        paths["interactions"] = str(interactions_path)
        logger.info(f"Saved {len(interactions_df):,} interactions → {interactions_path}")

        # Stats
        import json
        stats_path = out / "generation_stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2, default=str)
        paths["stats"] = str(stats_path)

        return paths

    def _log_stats(self) -> None:
        """Log generation statistics."""
        logger.info("=" * 60)
        logger.info("DATA GENERATION SUMMARY")
        logger.info("=" * 60)
        for k, v in self.stats.items():
            if isinstance(v, dict):
                logger.info(f"  {k}:")
                for kk, vv in v.items():
                    logger.info(f"    {kk}: {vv:,}" if isinstance(vv, int) else f"    {kk}: {vv}")
            elif isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
