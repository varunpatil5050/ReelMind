import numpy as np
import pytest
import torch
import faiss

from services.retrieval.two_tower import TwoTowerModel, TwoTowerInference
from services.retrieval.faiss_index import FAISSIndex

def test_two_tower_forward():
    model = TwoTowerModel(user_feature_dim=13, video_feature_dim=17, embedding_dim=128)
    u_feat = torch.rand(4, 13)
    v_feat = torch.rand(4, 17)
    
    # Check embedding shapes
    u_emb = model.get_user_embedding(u_feat)
    v_emb = model.get_video_embedding(v_feat)
    assert u_emb.shape == (4, 128)
    assert v_emb.shape == (4, 128)
    
    # Check L2 normalization (norm should be ~1.0)
    assert torch.allclose(torch.norm(u_emb, p=2, dim=1), torch.ones(4), atol=1e-5)
    
    # Check forward output (logits)
    logits = model(u_feat, v_feat)
    assert logits.shape == (4, 4)

def test_inference_wrapper():
    inference = TwoTowerInference()
    u_feat = [0.1] * 13
    v_feat = [0.2] * 17
    
    u_emb = inference.get_user_embedding(u_feat)
    v_emb = inference.get_video_embedding(v_feat)
    
    assert len(u_emb) == 128
    assert len(v_emb) == 128

def test_faiss_index():
    index = FAISSIndex(embedding_dim=128)
    
    # Generate dummy embeddings
    np.random.seed(42)
    embeddings = np.random.randn(100, 128).astype(np.float32)
    # L2 normalize
    faiss.normalize_L2(embeddings)
    video_ids = [f"v_{i}" for i in range(100)]
    
    # Build
    index.build_index(video_ids, embeddings, nlist=1, use_pq=False)
    
    # Search
    query = embeddings[0:1]  # Use first embedding as query
    results = index.search(query, top_k=5)
    
    assert len(results) == 5
    # The first result should be exactly the query itself (dist ~ 1.0 since it's inner product of normalized vectors)
    assert results[0][0] == "v_0"
    assert results[0][1] > 0.99
