"""
Two-Tower Retrieval Model for Candidate Generation.

The Two-Tower architecture maps users and videos into the same 128-dimensional
embedding space. The dot product (or cosine similarity) between a user
embedding and a video embedding indicates relevance.

Architecture:
- User Tower: MLP over user features (demographics, engagement stats)
- Video Tower: MLP over video features (content stats, creator stats)
- Output: 128d L2-normalized embeddings for both.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        user_feature_dim: int = 13,  # From UserFeatures.to_vector()
        video_feature_dim: int = 17, # From VideoFeatures.to_vector()
        embedding_dim: int = 128,
        hidden_dims: list[int] = [256, 128],
    ):
        super().__init__()
        
        # User Tower
        user_layers = []
        in_dim = user_feature_dim
        for h_dim in hidden_dims:
            user_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_dim = h_dim
        user_layers.append(nn.Linear(in_dim, embedding_dim))
        self.user_tower = nn.Sequential(*user_layers)
        
        # Video Tower
        video_layers = []
        in_dim = video_feature_dim
        for h_dim in hidden_dims:
            video_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_dim = h_dim
        video_layers.append(nn.Linear(in_dim, embedding_dim))
        self.video_tower = nn.Sequential(*video_layers)
        
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def get_user_embedding(self, user_features: torch.Tensor) -> torch.Tensor:
        """Forward pass for user tower."""
        embeds = self.user_tower(user_features)
        return F.normalize(embeds, p=2, dim=1)
        
    def get_video_embedding(self, video_features: torch.Tensor) -> torch.Tensor:
        """Forward pass for video tower."""
        embeds = self.video_tower(video_features)
        return F.normalize(embeds, p=2, dim=1)

    def forward(self, user_features: torch.Tensor, video_features: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity scores between users and videos.
        Returns: logits (batch_size, batch_size) if computing in-batch negatives
        """
        u_emb = self.get_user_embedding(user_features)
        v_emb = self.get_video_embedding(video_features)
        
        # Cosine similarity (since embeddings are L2 normalized)
        # Scaled by temperature
        logits = torch.matmul(u_emb, v_emb.T) / torch.exp(self.temperature)
        return logits


class TwoTowerInference:
    """Wrapper for running inference on the Two-Tower model."""
    
    def __init__(self, model_path: str | None = None):
        self.model = TwoTowerModel()
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        
    def get_user_embedding(self, user_vector: list[float]) -> list[float]:
        """Get 128d embedding for a single user."""
        with torch.no_grad():
            tensor = torch.tensor([user_vector], dtype=torch.float32)
            emb = self.model.get_user_embedding(tensor)
            return emb[0].tolist()
            
    def get_video_embedding(self, video_vector: list[float]) -> list[float]:
        """Get 128d embedding for a single video."""
        with torch.no_grad():
            tensor = torch.tensor([video_vector], dtype=torch.float32)
            emb = self.model.get_video_embedding(tensor)
            return emb[0].tolist()
