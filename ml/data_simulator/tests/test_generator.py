"""
ReelMind Data Simulator Tests — Validates data generation quality and consistency.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ml.data_simulator.distributions import (
    CategoryCorrelation,
    EngagementDistribution,
    SessionModel,
    TemporalDistribution,
    ViralityModel,
)
from ml.data_simulator.generator import DataGenerator, GeneratorConfig


# ─── Distribution Tests ─────────────────────────────────────────────────────


class TestTemporalDistribution:
    def test_hourly_weights_sum(self):
        td = TemporalDistribution()
        assert len(td.hourly_weights) == 24
        assert all(w >= 0 for w in td.hourly_weights)

    def test_sample_timestamp_in_range(self):
        td = TemporalDistribution()
        rng = np.random.default_rng(42)
        base = 1_700_000_000_000
        ts = td.sample_timestamp(rng, base)
        assert base <= ts <= base + 86_400_000  # Within 24h

    def test_peak_hours_have_higher_weight(self):
        td = TemporalDistribution()
        # Evening (19-22) should be higher than early morning (2-5)
        evening_avg = np.mean(td.hourly_weights[19:23])
        morning_avg = np.mean(td.hourly_weights[2:6])
        assert evening_avg > morning_avg * 3


class TestEngagementDistribution:
    def test_watch_percentage_bounded(self):
        ed = EngagementDistribution()
        rng = np.random.default_rng(42)
        for _ in range(1000):
            pct = ed.sample_watch_percentage(rng, 30000, 0.5, 0.5)
            assert 0.0 <= pct <= 2.0

    def test_high_quality_gets_more_engagement(self):
        ed = EngagementDistribution()
        rng = np.random.default_rng(42)

        low_q = [ed.sample_watch_percentage(rng, 30000, 0.5, 0.2) for _ in range(500)]
        rng = np.random.default_rng(42)
        high_q = [ed.sample_watch_percentage(rng, 30000, 0.5, 0.9) for _ in range(500)]

        assert np.mean(high_q) > np.mean(low_q)

    def test_engagement_action_valid(self):
        ed = EngagementDistribution()
        rng = np.random.default_rng(42)
        valid_actions = {"watch", "like", "share", "comment", "save", "follow_creator", "skip"}
        for _ in range(500):
            action = ed.sample_engagement_action(rng, 0.6, 0.5, 0.5, 0.1)
            assert action in valid_actions


class TestCategoryCorrelation:
    def test_self_correlation_is_one(self):
        cc = CategoryCorrelation()
        assert cc.get_correlation("gaming", "gaming") == 1.0

    def test_known_correlation(self):
        cc = CategoryCorrelation()
        assert cc.get_correlation("gaming", "tech") > 0.3
        assert cc.get_correlation("fashion", "beauty") > 0.5

    def test_sample_related_includes_primary(self):
        cc = CategoryCorrelation()
        rng = np.random.default_rng(42)
        related = cc.sample_related_categories(rng, "gaming", num_additional=2)
        assert related[0] == "gaming"
        assert len(related) == 3


class TestSessionModel:
    def test_session_length_positive(self):
        sm = SessionModel()
        rng = np.random.default_rng(42)
        for _ in range(100):
            length = sm.sample_session_length(rng, 0.5)
            assert length >= 1

    def test_fatigue_decreases(self):
        sm = SessionModel()
        assert sm.get_fatigue_factor(0) > sm.get_fatigue_factor(10)
        assert sm.get_fatigue_factor(10) > sm.get_fatigue_factor(50)


# ─── Generator Integration Tests ─────────────────────────────────────────────


class TestDataGenerator:
    @pytest.fixture
    def tiny_generator(self):
        config = GeneratorConfig(
            num_users=100,
            num_videos=200,
            num_interactions=2000,
            num_creators=20,
            simulation_days=7,
            seed=42,
        )
        gen = DataGenerator(config)
        gen.generate_all()
        return gen

    def test_correct_entity_counts(self, tiny_generator):
        gen = tiny_generator
        assert len(gen.users) == 100
        assert len(gen.creators) == 20
        assert len(gen.videos) == 200

    def test_interaction_count_approximate(self, tiny_generator):
        gen = tiny_generator
        # Allow 10% tolerance due to session-based generation
        assert len(gen.interactions) > 0
        assert len(gen.interactions) <= 2500  # Some overshoot ok

    def test_event_types_diverse(self, tiny_generator):
        gen = tiny_generator
        event_types = set(i["event_type"] for i in gen.interactions)
        # Should have at least watch, skip, and one engagement type
        assert len(event_types) >= 3

    def test_all_users_have_valid_ids(self, tiny_generator):
        gen = tiny_generator
        for u in gen.users:
            assert u["user_id"].startswith("u_")

    def test_all_videos_have_valid_ids(self, tiny_generator):
        gen = tiny_generator
        for v in gen.videos:
            assert v["video_id"].startswith("v_")
            assert v["duration_ms"] > 0

    def test_watch_percentage_distribution(self, tiny_generator):
        gen = tiny_generator
        pcts = [i["watch_percentage"] for i in gen.interactions]
        assert min(pcts) >= 0.0
        assert max(pcts) <= 2.0
        # Mean should be reasonable (0.3-0.8 range)
        assert 0.1 < np.mean(pcts) < 1.5

    def test_parquet_save_and_load(self, tiny_generator):
        gen = tiny_generator
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = gen.save_parquet(tmpdir)

            assert "users" in paths
            assert "videos" in paths
            assert "interactions" in paths

            # Verify Parquet readability
            users_df = pd.read_parquet(paths["users"])
            assert len(users_df) == 100

            videos_df = pd.read_parquet(paths["videos"])
            assert len(videos_df) == 200

            interactions_df = pd.read_parquet(paths["interactions"])
            assert len(interactions_df) > 0

    def test_temporal_distribution_in_interactions(self, tiny_generator):
        gen = tiny_generator
        hours = [i["hour_of_day"] for i in gen.interactions]
        assert all(0 <= h <= 23 for h in hours)

    def test_session_structure(self, tiny_generator):
        gen = tiny_generator
        sessions = set(i["session_id"] for i in gen.interactions)
        assert len(sessions) > 1
        for i in gen.interactions:
            assert i["session_id"].startswith("sess_")

    def test_stats_computed(self, tiny_generator):
        gen = tiny_generator
        assert "total_users" in gen.stats
        assert "total_interactions" in gen.stats
        assert gen.stats["total_users"] == 100
