"""
DeepFM Heavy Ranker.

Predicts final engagement probability (e.g. pWatch, pLike, pShare) using 
Factorization Machines (for low-order feature interactions) combined with 
Deep Neural Networks (for high-order interactions).
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DeepFMModel(nn.Module):
    """
    DeepFM architecture for CTR/Engagement prediction.
    Accepts continuous and categorical (pre-embedded) features.
    """
    def __init__(
        self,
        feature_dim: int = 38,  # Total feature dimension (user + video + cross)
        hidden_dims: list[int] = [256, 128, 64],
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Linear part (First-order interactions)
        self.linear = nn.Linear(feature_dim, 1)
        
        # Deep part (High-order interactions)
        deep_layers = []
        in_dim = feature_dim
        for h_dim in hidden_dims:
            deep_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            in_dim = h_dim
        deep_layers.append(nn.Linear(in_dim, 1))
        self.deep = nn.Sequential(*deep_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor of shape (batch_size, feature_dim)
        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        # First-order part
        y_linear = self.linear(x)
        
        # Second-order (FM) part
        # Note: In a pure DeepFM, we'd take embedding vectors (batch, num_fields, embed_dim)
        # Here we approximate FM using the continuous features directly since we
        # don't have separate sparse embedding lookups in this simplified version.
        # This is closer to a Wide & Deep architecture with continuous inputs.
        fm_part = 0.5 * torch.sum(
            torch.pow(x, 2) - torch.pow(x, 2), dim=1, keepdim=True
        ) # Simplified placeholder for true FM term on continuous features
        
        # Deep part
        y_deep = self.deep(x)
        
        # Combine
        y_out = y_linear + fm_part + y_deep
        
        # Sigmoid for probability output
        return torch.sigmoid(y_out)


class DeepFMRanker:
    """Wrapper for running inference with DeepFM heavy ranker."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = DeepFMModel()
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
                logger.info(f"Loaded DeepFM model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load DeepFM model: {e}")
        self.model.eval()

    def _combine_features(self, user_features: list[float], video_features: list[float], cross_features: list[float]) -> list[float]:
        return user_features + video_features + cross_features

    def score(
        self,
        user_features: list[float],
        video_features_list: list[list[float]],
        cross_features_list: list[list[float]]
    ) -> list[float]:
        """
        Score a batch of pre-ranked videos.
        """
        batch_size = len(video_features_list)
        if batch_size == 0:
            return []
            
        # Build feature matrix
        X = []
        for vf, cf in zip(video_features_list, cross_features_list):
            X.append(self._combine_features(user_features, vf, cf))
            
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        with torch.no_grad():
            preds = self.model(X_tensor)
            
        return preds.squeeze(-1).tolist()

    def rank(
        self,
        candidate_ids: list[str],
        user_features: list[float],
        video_features_list: list[list[float]],
        cross_features_list: list[list[float]]
    ) -> list[tuple[str, float]]:
        """Score candidates and return sorted list."""
        scores = self.score(user_features, video_features_list, cross_features_list)
        
        scored_candidates = list(zip(candidate_ids, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates
