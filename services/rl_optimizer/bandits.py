"""
Reinforcement Learning Re-Ranker.

Uses Contextual Bandits to optimize the final feed ordering.
Balances exploiting high-engagement content with exploring new/niche content.
Enforces diversity constraints using Maximal Marginal Relevance (MMR).
"""

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ThompsonSamplingBandit:
    """
    Beta-Bernoulli Thompson Sampling for Contextual Bandits.
    Maintains beta distributions (alpha, beta) for different content categories.
    """
    def __init__(self):
        # Fallback parameters if we don't have database connection
        # Dict of category -> {alpha, beta}
        self.category_params = {}
        
    def _get_params(self, category: str) -> tuple[float, float]:
        """Get (alpha, beta) for a category, with prior (2.0, 10.0)."""
        if category not in self.category_params:
            self.category_params[category] = {"alpha": 2.0, "beta": 10.0}
        p = self.category_params[category]
        return p["alpha"], p["beta"]

    def sample_explore_score(self, category: str) -> float:
        """Sample from the Beta distribution for this category."""
        alpha, beta = self._get_params(category)
        # Sample from beta distribution
        sample = np.random.beta(alpha, beta)
        return float(sample)
        
    def update(self, category: str, reward: float):
        """
        Update the posterior distribution with new evidence.
        Reward should be 0.0 to 1.0 (e.g. watch percentage or CTR).
        """
        alpha, beta = self._get_params(category)
        
        # Exponential moving average style update to handle non-stationary preferences
        learning_rate = 0.05
        
        new_alpha = alpha * (1 - learning_rate) + reward * learning_rate * 10
        new_beta = beta * (1 - learning_rate) + (1 - reward) * learning_rate * 10
        
        # Keep parameters bounded
        self.category_params[category]["alpha"] = max(1.0, min(new_alpha, 100.0))
        self.category_params[category]["beta"] = max(1.0, min(new_beta, 100.0))


class DiversityReranker:
    """
    Maximal Marginal Relevance (MMR) for enforcing diversity.
    """
    def rerank(self, candidates: list[dict], lambda_param: float = 0.7) -> list[dict]:
        """
        Args:
            candidates: List of dicts with 'video_id', 'score', 'category'
            lambda_param: 1.0 = purely exploit scores, 0.0 = purely diverse
            
        Returns:
            Re-ordered list of candidates
        """
        if not candidates:
            return []
            
        # Copy to avoid modifying original
        unselected = list(candidates)
        selected = []
        
        # The first item is always the one with highest raw score
        unselected.sort(key=lambda x: x["score"], reverse=True)
        selected.append(unselected.pop(0))
        
        # Iteratively select the next item
        while unselected:
            best_score = -float('inf')
            best_idx = 0
            
            # Categories we've already selected
            selected_cats = [c["category"] for c in selected]
            
            for i, cand in enumerate(unselected):
                # Similarity to already selected items (simple categorical penalty)
                # In a real system, this would use embedding cosine similarity
                cat = cand["category"]
                cat_count = selected_cats.count(cat)
                
                # Penalty increases quadratically with count of same category
                penalty = min(cat_count * 0.15, 0.8) 
                
                # MMR Equation
                mmr_score = lambda_param * cand["score"] - (1 - lambda_param) * penalty
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
                    
            selected.append(unselected.pop(best_idx))
            
        return selected


class BanditReranker:
    """
    Combines Model Scores, Bandit Exploration, and Diversity MMR.
    """
    def __init__(self):
        self.ts_bandit = ThompsonSamplingBandit()
        self.diversity = DiversityReranker()
        
    def optimize_feed(self, ranked_candidates: list[dict], user_features: dict, num_results: int = 15) -> list[dict]:
        """
        Takes the heavy-ranked candidates and produces the final feed.
        """
        exploration_weight = 0.15  # How much TS score influences final order
        
        scored = []
        for cand in ranked_candidates:
            # We need the category for bandit. Assume it's passed or fetch it.
            # Here we expect cand to have 'category' injected by previous stage
            cat = cand.get("category", "unknown")
            
            # 1. Base score from DeepFM
            base_score = cand["score"]
            
            # 2. Add exploration term from Thompson Sampling
            explore_score = self.ts_bandit.sample_explore_score(cat)
            
            # Combined score
            final_score = base_score * (1 - exploration_weight) + explore_score * exploration_weight
            
            cand_copy = dict(cand)
            cand_copy["rl_score"] = final_score
            scored.append(cand_copy)
            
        # 3. Apply Diversity (MMR)
        diverse_feed = self.diversity.rerank(scored, lambda_param=0.7)
        
        return diverse_feed[:num_results]
