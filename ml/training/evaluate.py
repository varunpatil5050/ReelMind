"""
Offline Evaluation Metrics for Recommender Systems.
"""

import numpy as np

def calculate_mrr(ranked_lists: list[list[str]], relevant_items: list[set[str]]) -> float:
    """Mean Reciprocal Rank."""
    rr_sum = 0.0
    for ranked, relevant in zip(ranked_lists, relevant_items):
        for i, item in enumerate(ranked):
            if item in relevant:
                rr_sum += 1.0 / (i + 1)
                break
    return rr_sum / len(ranked_lists) if ranked_lists else 0.0

def calculate_recall_at_k(ranked_lists: list[list[str]], relevant_items: list[set[str]], k: int = 100) -> float:
    """Recall @ K."""
    recall_sum = 0.0
    for ranked, relevant in zip(ranked_lists, relevant_items):
        if not relevant:
            continue
        hits = sum(1 for item in ranked[:k] if item in relevant)
        recall_sum += hits / len(relevant)
    return recall_sum / len(ranked_lists) if ranked_lists else 0.0

def calculate_ndcg_at_k(ranked_lists: list[list[str]], relevant_items: list[set[str]], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain @ K."""
    ndcg_sum = 0.0
    for ranked, relevant in zip(ranked_lists, relevant_items):
        if not relevant:
            continue
            
        dcg = 0.0
        for i, item in enumerate(ranked[:k]):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 2)
                
        # Ideal DCG (all relevant items at the top)
        idcg = 0.0
        ideal_hits = min(len(relevant), k)
        for i in range(ideal_hits):
            idcg += 1.0 / np.log2(i + 2)
            
        if idcg > 0:
            ndcg_sum += dcg / idcg
            
    return ndcg_sum / len(ranked_lists) if ranked_lists else 0.0

def evaluate_retrieval(model, dataloader, k: int = 100):
    """Placeholder for full offline retrieval evaluation."""
    pass

def evaluate_ranking(model, dataloader):
    """Placeholder for full offline ranking evaluation (AUC, LogLoss)."""
    pass
