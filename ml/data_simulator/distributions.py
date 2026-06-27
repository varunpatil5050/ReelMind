"""
ReelMind Statistical Distributions — Models realistic user behavior patterns.

This module captures the nuanced statistical properties of short-video platforms:
- Power-law engagement distributions (most users are passive)
- Temporal cyclicality (usage peaks at evening, weekends)
- Category preference correlations (gaming↔tech, fashion↔beauty)
- Viral content dynamics (exponential early growth → plateau)
- Session behavior (geometric session lengths, binge patterns)

Every distribution is parameterized and documented to enable
sensitivity analysis and A/B test simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TemporalDistribution:
    """
    Models time-of-day and day-of-week usage patterns.
    
    Based on empirical patterns from short-video platforms:
    - Peak hours: 19:00-23:00 local time
    - Secondary peak: 12:00-13:00 (lunch)
    - Weekend boost: ~1.3x weekday traffic
    """

    # Hour weights (0-23), unnormalized
    hourly_weights: tuple[float, ...] = (
        0.15, 0.08, 0.05, 0.03, 0.03, 0.05,  # 00-05: late night/early morning
        0.10, 0.18, 0.25, 0.30, 0.35, 0.45,  # 06-11: morning ramp
        0.55, 0.50, 0.40, 0.38, 0.42, 0.55,  # 12-17: afternoon
        0.70, 0.85, 0.95, 1.00, 0.80, 0.50,  # 18-23: evening peak
    )

    # Day-of-week multipliers (Mon=0, Sun=6)
    daily_multipliers: tuple[float, ...] = (
        0.85, 0.85, 0.90, 0.90, 1.00, 1.25, 1.20
    )

    def sample_timestamp(
        self,
        rng: np.random.Generator,
        base_timestamp_ms: int,
        window_hours: int = 24,
    ) -> int:
        """Sample a timestamp weighted by temporal patterns."""
        hour_probs = np.array(self.hourly_weights)
        hour_probs = hour_probs / hour_probs.sum()
        hour = rng.choice(24, p=hour_probs)
        minute = rng.integers(0, 60)
        second = rng.integers(0, 60)
        offset_ms = (hour * 3600 + minute * 60 + second) * 1000
        return base_timestamp_ms + offset_ms

    def get_activity_multiplier(self, hour: int, day_of_week: int) -> float:
        """Get combined activity multiplier for a given time."""
        return self.hourly_weights[hour] * self.daily_multipliers[day_of_week]


@dataclass(frozen=True)
class EngagementDistribution:
    """
    Models engagement patterns with realistic power-law characteristics.
    
    Key properties:
    - Watch time follows log-normal (heavy right tail for viral content)
    - Skip probability decreases with watch percentage (commitment effect)
    - Like/share/comment follow power-law (most videos get few interactions)
    - Replay probability is bimodal (either 0 or multiple replays)
    """

    # Watch time distribution parameters
    watch_time_mu: float = 2.5      # log-normal mean (in log-seconds)
    watch_time_sigma: float = 0.8   # log-normal std

    # Engagement probability baselines (per-impression)
    base_like_prob: float = 0.045
    base_share_prob: float = 0.008
    base_comment_prob: float = 0.012
    base_save_prob: float = 0.015
    base_skip_prob: float = 0.35
    base_follow_prob: float = 0.003

    # Replay parameters
    replay_prob: float = 0.08       # Probability of any replay
    replay_count_lambda: float = 1.5  # Poisson parameter for replay count

    def sample_watch_percentage(
        self,
        rng: np.random.Generator,
        video_duration_ms: int,
        user_engagement_level: float,
        content_quality: float,
    ) -> float:
        """
        Sample watch percentage considering user engagement and content quality.
        
        Returns 0.0-2.0 (>1.0 indicates replay/rewatch).
        """
        # Base watch from log-normal
        base = rng.lognormal(self.watch_time_mu, self.watch_time_sigma)
        base_pct = min(base * 1000 / max(video_duration_ms, 1000), 1.0)

        # User engagement modulates completion probability
        engagement_boost = 0.7 + 0.6 * user_engagement_level
        quality_boost = 0.6 + 0.8 * content_quality

        watch_pct = base_pct * engagement_boost * quality_boost

        # Bimodal replay: some users rewatch good content
        if watch_pct > 0.85 and rng.random() < self.replay_prob * content_quality:
            replays = rng.poisson(self.replay_count_lambda)
            watch_pct = min(1.0 + replays * rng.uniform(0.3, 1.0), 2.0)

        return min(max(watch_pct, 0.0), 2.0)

    def sample_engagement_action(
        self,
        rng: np.random.Generator,
        watch_percentage: float,
        user_engagement_level: float,
        content_quality: float,
        virality_score: float,
    ) -> str:
        """
        Sample what engagement action happens after watching.
        
        Actions are not mutually exclusive in reality, but for event
        generation we sample the primary action. The training pipeline
        can use multi-label targets.
        """
        # Scale probabilities by watch commitment and quality
        commitment = min(watch_percentage / 0.8, 1.5)
        quality_mult = 0.5 + content_quality
        user_mult = 0.3 + 1.4 * user_engagement_level
        viral_mult = 1.0 + 2.0 * virality_score

        # Skip is inversely related to commitment
        skip_prob = self.base_skip_prob * (1.5 - commitment)

        if watch_percentage < 0.15:
            return "skip"

        if rng.random() < skip_prob:
            return "skip"

        # Sample positive engagement
        probs = {
            "watch": 1.0,  # Default: just watched, no explicit action
            "like": self.base_like_prob * commitment * quality_mult * user_mult,
            "share": self.base_share_prob * commitment * quality_mult * viral_mult,
            "comment": self.base_comment_prob * commitment * user_mult,
            "save": self.base_save_prob * commitment * quality_mult,
            "follow_creator": self.base_follow_prob * commitment * quality_mult,
        }

        actions = list(probs.keys())
        weights = np.array(list(probs.values()))
        weights = weights / weights.sum()

        return rng.choice(actions, p=weights)


@dataclass(frozen=True)
class CategoryCorrelation:
    """
    Models category preference correlations.
    
    Users who like gaming also tend to like tech;
    fashion enthusiasts often also follow beauty content.
    This drives realistic multi-interest user profiles.
    """

    # Correlation matrix (symmetric, diagonal = 1.0)
    # Categories indexed by ContentCategory enum order
    _correlations: dict[tuple[str, str], float] = field(default_factory=lambda: {
        ("comedy", "music"): 0.4,
        ("comedy", "pets"): 0.3,
        ("dance", "music"): 0.7,
        ("dance", "fashion"): 0.4,
        ("education", "tech"): 0.5,
        ("education", "news"): 0.4,
        ("food", "travel"): 0.5,
        ("food", "diy"): 0.3,
        ("gaming", "tech"): 0.6,
        ("gaming", "comedy"): 0.3,
        ("music", "dance"): 0.7,
        ("music", "fashion"): 0.3,
        ("sports", "fitness"): 0.6,
        ("tech", "education"): 0.5,
        ("tech", "gaming"): 0.6,
        ("fashion", "beauty"): 0.7,
        ("fashion", "dance"): 0.4,
        ("travel", "food"): 0.5,
        ("travel", "photography"): 0.4,
        ("fitness", "sports"): 0.6,
        ("fitness", "food"): 0.3,
        ("beauty", "fashion"): 0.7,
        ("diy", "food"): 0.3,
        ("news", "education"): 0.4,
    })

    def get_correlation(self, cat_a: str, cat_b: str) -> float:
        if cat_a == cat_b:
            return 1.0
        key = (cat_a, cat_b)
        rev_key = (cat_b, cat_a)
        return self._correlations.get(key, self._correlations.get(rev_key, 0.05))

    def sample_related_categories(
        self,
        rng: np.random.Generator,
        primary_category: str,
        num_additional: int = 2,
        all_categories: Optional[list[str]] = None,
    ) -> list[str]:
        """Sample additional categories correlated with the primary one."""
        if all_categories is None:
            all_categories = [
                "comedy", "dance", "education", "food", "gaming",
                "music", "sports", "tech", "fashion", "travel",
                "fitness", "pets", "news", "diy", "beauty",
            ]

        other_cats = [c for c in all_categories if c != primary_category]
        weights = np.array([
            self.get_correlation(primary_category, c) for c in other_cats
        ])
        weights = weights / weights.sum()

        chosen = rng.choice(
            other_cats,
            size=min(num_additional, len(other_cats)),
            replace=False,
            p=weights,
        )
        return [primary_category] + list(chosen)


@dataclass
class ViralityModel:
    """
    Models viral content dynamics.
    
    Viral spread follows a modified logistic curve:
    - Exponential early growth (first 2-6 hours)
    - Inflection point at ~10-20% of peak
    - Plateau and slow decay
    
    Only ~2-5% of content goes viral.
    """

    viral_probability: float = 0.03
    peak_multiplier_range: tuple[float, float] = (5.0, 50.0)
    growth_rate_range: tuple[float, float] = (0.5, 2.0)
    peak_hour_range: tuple[int, int] = (4, 48)

    def is_viral(self, rng: np.random.Generator, content_quality: float) -> bool:
        """Determine if content goes viral (quality-weighted)."""
        adjusted_prob = self.viral_probability * (0.5 + content_quality)
        return rng.random() < adjusted_prob

    def get_viral_multiplier(
        self,
        rng: np.random.Generator,
        hours_since_upload: float,
    ) -> float:
        """
        Get engagement multiplier based on viral curve position.
        
        Uses logistic growth: M / (1 + exp(-k*(t - t_peak)))
        """
        peak_mult = rng.uniform(*self.peak_multiplier_range)
        growth_rate = rng.uniform(*self.growth_rate_range)
        peak_hour = rng.uniform(*self.peak_hour_range)

        # Logistic growth curve
        if hours_since_upload < 0:
            return 1.0

        growth = peak_mult / (1 + math.exp(-growth_rate * (hours_since_upload - peak_hour)))

        # Decay after peak (slow exponential decay)
        if hours_since_upload > peak_hour * 2:
            decay = math.exp(-0.02 * (hours_since_upload - peak_hour * 2))
            growth *= decay

        return max(growth, 1.0)


@dataclass(frozen=True)
class SessionModel:
    """
    Models user session behavior.
    
    Key patterns:
    - Session length follows negative binomial (heavy tail)
    - Inter-session gap follows exponential
    - Binge sessions (~5% of sessions) are 3-5x longer
    - Session engagement decays over time (fatigue)
    """

    mean_session_videos: float = 12.0
    session_variance: float = 8.0
    binge_probability: float = 0.05
    binge_multiplier: float = 3.5
    fatigue_decay_rate: float = 0.02  # Engagement drops per video in session

    def sample_session_length(
        self,
        rng: np.random.Generator,
        user_engagement_level: float,
    ) -> int:
        """Sample number of videos in a session."""
        mean = self.mean_session_videos * (0.5 + user_engagement_level)

        # Check for binge session
        is_binge = rng.random() < self.binge_probability
        if is_binge:
            mean *= self.binge_multiplier

        # Negative binomial for heavy-tailed session lengths
        n = max(mean ** 2 / max(self.session_variance - mean, 1), 1)
        p = n / (n + mean)
        length = rng.negative_binomial(max(int(n), 1), min(max(p, 0.01), 0.99))

        return max(length, 1)

    def get_fatigue_factor(self, position_in_session: int) -> float:
        """Get engagement multiplier based on session fatigue."""
        return math.exp(-self.fatigue_decay_rate * position_in_session)


@dataclass(frozen=True)
class ColdStartModel:
    """
    Models cold-start behavior for new users and new content.
    
    New users:
    - Higher exploration (try diverse categories)
    - Lower engagement initially
    - Engagement ramps up over first ~50 interactions
    
    New content:
    - Shown to small "test" audience first
    - Quality signal aggregated from early viewers
    - Promoted if early engagement is high
    """

    user_warmup_interactions: int = 50
    content_test_audience_size: int = 100
    new_user_exploration_boost: float = 2.0
    new_content_quality_uncertainty: float = 0.3

    def get_user_maturity(self, num_interactions: int) -> float:
        """Returns 0.0 (brand new) to 1.0 (mature user)."""
        return min(num_interactions / self.user_warmup_interactions, 1.0)

    def get_content_confidence(self, num_impressions: int) -> float:
        """Returns confidence in content quality estimate."""
        return min(num_impressions / self.content_test_audience_size, 1.0)
