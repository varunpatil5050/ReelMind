"""
LightGBM Pre-Ranker.

Filters candidates from retrieval (e.g., 500 down to 50) using a fast
tree-based model. Computes scores using user stats, video stats, and 
cross-features.
"""

import logging
from typing import Optional

import lightgbm as lgb
import numpy as np

logger = logging.getLogger(__name__)


class LightGBMPreRanker:
    """Wrapper for running inference with LightGBM pre-ranker."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path:
            try:
                self.model = lgb.Booster(model_file=model_path)
                logger.info(f"Loaded LightGBM model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load LightGBM model: {e}")
                
    def _combine_features(self, user_features: list[float], video_features: list[float], cross_features: list[float]) -> list[float]:
        """Combine feature vectors in the order expected by the model."""
        # Expected dim: 13 (user) + 17 (video) + 8 (cross) = 38
        return user_features + video_features + cross_features

    def score(
        self,
        user_features: list[float],
        video_features_list: list[list[float]],
        cross_features_list: list[list[float]]
    ) -> list[float]:
        """
        Score a batch of videos for a given user.
        
        Args:
            user_features: 13d vector
            video_features_list: list of 17d vectors
            cross_features_list: list of 8d vectors
            
        Returns:
            List of float scores (predicted watch percentage)
        """
        batch_size = len(video_features_list)
        if batch_size == 0:
            return []
            
        assert len(video_features_list) == len(cross_features_list)
        
        if self.model is None:
            # Fallback random scorer if no model loaded
            # Use simple heuristic: sum of normalized cross features + random
            scores = []
            for cf in cross_features_list:
                base = sum(cf) / len(cf)
                scores.append(base + np.random.uniform(0.0, 0.2))
            return scores
            
        # Build feature matrix
        X = []
        for vf, cf in zip(video_features_list, cross_features_list):
            X.append(self._combine_features(user_features, vf, cf))
            
        X_np = np.array(X, dtype=np.float32)
        
        # Predict
        preds = self.model.predict(X_np)
        return preds.tolist()

    def rerank(
        self,
        candidate_ids: list[str],
        user_features: list[float],
        video_features_list: list[list[float]],
        cross_features_list: list[list[float]],
        top_k: int = 50
    ) -> list[tuple[str, float]]:
        """Score candidates and return top-K sorted."""
        scores = self.score(user_features, video_features_list, cross_features_list)
        
        # Pair with IDs and sort descending
        scored_candidates = list(zip(candidate_ids, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates[:top_k]
