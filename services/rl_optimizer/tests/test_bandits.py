import pytest

from services.rl_optimizer.bandits import ThompsonSamplingBandit, DiversityReranker, BanditReranker

def test_thompson_sampling():
    bandit = ThompsonSamplingBandit()
    
    # Initial state
    a1, b1 = bandit._get_params("comedy")
    assert a1 == 2.0
    assert b1 == 10.0
    
    # Update with high reward
    bandit.update("comedy", 1.0)
    a2, b2 = bandit._get_params("comedy")
    assert a2 > a1  # Alpha should increase
    assert b2 < b1  # Beta should decrease
    
    # Sample
    score = bandit.sample_explore_score("comedy")
    assert 0.0 <= score <= 1.0

def test_diversity_reranker():
    diversity = DiversityReranker()
    
    # Create candidates heavily skewed to 'gaming'
    candidates = [
        {"video_id": "v1", "score": 0.9, "category": "gaming"},
        {"video_id": "v2", "score": 0.85, "category": "gaming"},
        {"video_id": "v3", "score": 0.8, "category": "gaming"},
        {"video_id": "v4", "score": 0.7, "category": "comedy"},
    ]
    
    # With MMR (lambda=0.5), it should demote v2/v3 because they are same category as v1
    # Thus v4 (comedy) should jump up in rank
    reranked = diversity.rerank(candidates, lambda_param=0.5)
    
    assert len(reranked) == 4
    # First item is always highest score
    assert reranked[0]["video_id"] == "v1"
    # Second item should be v4 due to diversity penalty on gaming
    assert reranked[1]["video_id"] == "v4"

def test_bandit_reranker():
    reranker = BanditReranker()
    
    candidates = [
        {"video_id": "v1", "score": 0.9, "category": "gaming"},
        {"video_id": "v2", "score": 0.8, "category": "comedy"},
    ]
    
    # Should not crash, returns the right number of results
    feed = reranker.optimize_feed(candidates, {}, num_results=2)
    assert len(feed) == 2
    assert "rl_score" in feed[0]
