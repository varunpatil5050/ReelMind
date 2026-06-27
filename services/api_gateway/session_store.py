"""
In-Memory Session Store — Tracks user interactions for live personalization.

In production this would be Redis/DynamoDB. For local demo, in-memory is fine.
Tracks: likes, skips, watch times, category affinities, session aggregates.
"""

from __future__ import annotations

import time
import collections
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UserSession:
    """Tracks a single user's session state."""
    user_id: str
    started_at: float = field(default_factory=time.time)
    
    # Raw counters
    total_watches: int = 0
    total_likes: int = 0
    total_skips: int = 0
    total_shares: int = 0
    total_comments: int = 0
    
    # Watch time tracking
    watch_durations_ms: list[float] = field(default_factory=list)
    watch_percentages: list[float] = field(default_factory=list)
    
    # Category interaction tracking
    category_likes: dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))
    category_watches: dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))
    category_skips: dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))
    category_watch_time: dict[str, float] = field(default_factory=lambda: collections.defaultdict(float))
    
    # Last served feed (for explanations)
    last_feed: Optional[list[dict]] = None
    last_feed_timings: Optional[dict] = None
    last_feed_timestamp: float = 0.0
    
    # Interaction history (last N events)
    recent_events: list[dict] = field(default_factory=list)
    
    def record_event(self, event_type: str, video_id: str, category: str,
                     watch_duration_ms: float = 0, watch_percentage: float = 0):
        """Record a user interaction event."""
        
        event = {
            "event_type": event_type,
            "video_id": video_id,
            "category": category,
            "watch_duration_ms": watch_duration_ms,
            "watch_percentage": watch_percentage,
            "timestamp": time.time(),
        }
        self.recent_events.append(event)
        if len(self.recent_events) > 100:
            self.recent_events = self.recent_events[-100:]
        
        self.category_watches[category] += 1
        self.total_watches += 1
        
        if watch_duration_ms > 0:
            self.watch_durations_ms.append(watch_duration_ms)
        if watch_percentage > 0:
            self.watch_percentages.append(watch_percentage)
        
        if event_type == "like":
            self.total_likes += 1
            self.category_likes[category] += 1
            self.category_watch_time[category] += watch_duration_ms
        elif event_type == "skip":
            self.total_skips += 1
            self.category_skips[category] += 1
        elif event_type == "share":
            self.total_shares += 1
            self.category_watch_time[category] += watch_duration_ms
        elif event_type == "comment":
            self.total_comments += 1
            self.category_watch_time[category] += watch_duration_ms
        elif event_type == "watch":
            self.category_watch_time[category] += watch_duration_ms
    
    def get_category_affinities(self) -> dict[str, float]:
        """
        Compute category affinity scores (0.0 - 1.0) based on interactions.
        
        Affinity = weighted combination of:
          - Like rate per category (weight: 0.4)
          - Watch time per category (weight: 0.3)  
          - Inverse skip rate (weight: 0.3)
        """
        all_categories = set(
            list(self.category_watches.keys()) +
            list(self.category_likes.keys()) +
            list(self.category_skips.keys())
        )
        
        if not all_categories:
            return {}
        
        affinities = {}
        
        # Normalize across categories
        max_watches = max(self.category_watches.values()) if self.category_watches else 1
        max_watch_time = max(self.category_watch_time.values()) if self.category_watch_time else 1
        
        for cat in all_categories:
            watches = self.category_watches.get(cat, 0)
            likes = self.category_likes.get(cat, 0)
            skips = self.category_skips.get(cat, 0)
            watch_time = self.category_watch_time.get(cat, 0)
            
            # Like rate (0-1)
            like_rate = likes / max(watches, 1)
            
            # Normalized watch time (0-1)
            norm_watch_time = watch_time / max(max_watch_time, 1)
            
            # Inverse skip rate (1 = never skips, 0 = always skips)
            skip_rate = 1.0 - (skips / max(watches, 1))
            
            affinity = 0.4 * like_rate + 0.3 * norm_watch_time + 0.3 * max(skip_rate, 0)
            affinities[cat] = round(min(affinity, 1.0), 3)
        
        # Sort by affinity descending
        return dict(sorted(affinities.items(), key=lambda x: x[1], reverse=True))
    
    def get_session_stats(self) -> dict:
        """Return aggregate session statistics."""
        avg_watch_time = (
            sum(self.watch_durations_ms) / len(self.watch_durations_ms)
            if self.watch_durations_ms else 0
        )
        avg_watch_pct = (
            sum(self.watch_percentages) / len(self.watch_percentages)
            if self.watch_percentages else 0
        )
        
        # Retention score: weighted combo of watch completion and engagement
        engagement_rate = (self.total_likes + self.total_shares + self.total_comments) / max(self.total_watches, 1)
        retention_score = 0.6 * min(avg_watch_pct, 1.0) + 0.4 * min(engagement_rate, 1.0)
        
        return {
            "total_watches": self.total_watches,
            "total_likes": self.total_likes,
            "total_skips": self.total_skips,
            "total_shares": self.total_shares,
            "total_comments": self.total_comments,
            "avg_watch_time_ms": round(avg_watch_time, 1),
            "avg_watch_percentage": round(avg_watch_pct, 3),
            "retention_score": round(retention_score, 3),
            "engagement_rate": round(engagement_rate, 4),
            "session_duration_s": round(time.time() - self.started_at, 1),
        }
    
    def to_profile(self) -> dict:
        """Full user profile for the analytics dashboard."""
        return {
            "user_id": self.user_id,
            "session_stats": self.get_session_stats(),
            "category_affinities": self.get_category_affinities(),
            "recent_events": self.recent_events[-20:],  # Last 20
            "last_feed_timings": self.last_feed_timings,
            "last_feed": [
                {
                    "video_id": v.get("video_id"),
                    "score": v.get("score", 0),
                    "rank": v.get("rank", 0),
                    "retrieval_source": v.get("retrieval_source", "two_tower"),
                }
                for v in (self.last_feed or [])
            ],
        }


class SessionStore:
    """Global session store. In production → Redis."""
    
    def __init__(self):
        self._sessions: dict[str, UserSession] = {}
    
    def get_or_create(self, user_id: str) -> UserSession:
        if user_id not in self._sessions:
            self._sessions[user_id] = UserSession(user_id=user_id)
        return self._sessions[user_id]
    
    def get(self, user_id: str) -> Optional[UserSession]:
        return self._sessions.get(user_id)
    
    def all_users(self) -> list[str]:
        return list(self._sessions.keys())


# Global singleton
session_store = SessionStore()
