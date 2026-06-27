"""
ReelMind Dataset — Loads Parquet data for PyTorch/LightGBM training.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ReelMindDataset(Dataset):
    """
    PyTorch Dataset for ReelMind generated data.
    Loads interactions and joins with user/video features.
    """
    def __init__(self, data_dir: str, max_samples: int | None = None):
        self.data_dir = data_dir
        
        # Load interactions
        int_path = os.path.join(data_dir, "interactions.parquet")
        if not os.path.exists(int_path):
            raise FileNotFoundError(f"Interactions not found at {int_path}")
            
        self.interactions_df = pd.read_parquet(int_path)
        if max_samples:
            self.interactions_df = self.interactions_df.head(max_samples)
            
        # We simulate feature extraction by generating random features for now,
        # in a real pipeline this would join with feature store snapshot or read 
        # pre-computed offline features.
        
        # Determine labels: Positive engagement (watch > 50% or like/share/comment)
        # Note: in reality we'd have multi-task labels
        self.labels = []
        for _, row in self.interactions_df.iterrows():
            wp = row.get("watch_percentage", 0.0)
            et = row.get("event_type", "skip")
            if wp > 0.5 or et in ("like", "share", "comment", "save"):
                self.labels.append(1.0)
            else:
                self.labels.append(0.0)
                
    def __len__(self):
        return len(self.interactions_df)
        
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            user_features (13d)
            video_features (17d)
            cross_features (8d)
            label (1d)
        """
        # For simplicity in this demo dataset, we return synthetic features
        # since we didn't save feature store dumps in the generator.
        
        # Set seed based on row so it's deterministic but varies per row
        rng = np.random.default_rng(idx)
        
        u_feat = rng.random(13).astype(np.float32)
        v_feat = rng.random(17).astype(np.float32)
        c_feat = rng.random(8).astype(np.float32)
        label = np.array([self.labels[idx]], dtype=np.float32)
        
        return (
            torch.from_numpy(u_feat),
            torch.from_numpy(v_feat),
            torch.from_numpy(c_feat),
            torch.from_numpy(label)
        )
