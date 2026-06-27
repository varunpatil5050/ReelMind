import pytest
import torch

from services.ranking.preranker import LightGBMPreRanker
from services.ranking.deepfm import DeepFMRanker, DeepFMModel

def test_preranker_fallback():
    preranker = LightGBMPreRanker()
    u_feat = [0.1] * 13
    v_feat_list = [[0.2] * 17 for _ in range(5)]
    c_feat_list = [[0.3] * 8 for _ in range(5)]
    ids = [f"v_{i}" for i in range(5)]
    
    # Should work without a real model (uses fallback heuristic)
    results = preranker.rerank(ids, u_feat, v_feat_list, c_feat_list, top_k=3)
    
    assert len(results) == 3
    # Check that it returns (video_id, score) tuples
    assert isinstance(results[0][0], str)
    assert isinstance(results[0][1], float)

def test_deepfm_forward():
    model = DeepFMModel(feature_dim=38) # 13+17+8
    x = torch.rand(4, 38)
    
    y = model(x)
    assert y.shape == (4, 1)
    
    # Sigmoid output should be in [0, 1]
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)

def test_deepfm_ranker():
    ranker = DeepFMRanker()
    u_feat = [0.1] * 13
    v_feat_list = [[0.2] * 17 for _ in range(5)]
    c_feat_list = [[0.3] * 8 for _ in range(5)]
    ids = [f"v_{i}" for i in range(5)]
    
    results = ranker.rank(ids, u_feat, v_feat_list, c_feat_list)
    
    assert len(results) == 5
    # Results should be sorted by score descending
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
