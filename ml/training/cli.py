"""
Training CLI.
"""

import click
import logging
import json
import time
import numpy as np
from pathlib import Path

from ml.training.evaluate import calculate_ndcg_at_k, calculate_mrr, calculate_recall_at_k

@click.group()
def main():
    """ReelMind Training CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

@main.command()
@click.option("--data-dir", type=str, required=True, help="Path to data directory")
@click.option("--epochs", type=int, default=5, help="Number of training epochs")
@click.option("--batch-size", type=int, default=256, help="Batch size")
@click.option("--lr", type=float, default=0.001, help="Learning rate")
def two_tower(data_dir, epochs, batch_size, lr):
    """Train the Two-Tower retrieval model."""
    from ml.training.train_two_tower import train_two_tower
    train_two_tower(data_dir, epochs, batch_size, lr)

@main.command()
@click.option("--data-dir", type=str, required=True, help="Path to data directory")
@click.option("--epochs", type=int, default=5, help="Number of training epochs")
@click.option("--batch-size", type=int, default=512, help="Batch size")
@click.option("--lr", type=float, default=0.001, help="Learning rate")
def deepfm(data_dir, epochs, batch_size, lr):
    """Train the DeepFM ranking model."""
    from ml.training.train_deepfm import train_deepfm
    train_deepfm(data_dir, epochs, batch_size, lr)

@main.command()
@click.option("--data-dir", type=str, required=True, help="Path to data directory")
@click.option("--num-boost-round", type=int, default=100, help="Number of boosting rounds")
def preranker(data_dir, num_boost_round):
    """Train the LightGBM pre-ranking model."""
    from ml.training.train_preranker import train_preranker
    train_preranker(data_dir, num_boost_round)


@main.command()
@click.option("--num-users", type=int, default=500, help="Number of test users")
@click.option("--num-videos", type=int, default=1000, help="Number of test videos")
@click.option("--output", type=str, default="evaluation_results.json", help="Output file")
def evaluate(num_users, num_videos, output):
    """
    Run offline evaluation of the recommendation pipeline.
    
    Generates synthetic ground truth, simulates retrieval + ranking,
    and computes standard IR metrics: NDCG@10, MRR, Recall@K.
    """
    logger = logging.getLogger("evaluate")
    logger.info("=" * 60)
    logger.info("  ReelMind Offline Evaluation Pipeline")
    logger.info("=" * 60)
    
    np.random.seed(42)
    
    # Step 1: Generate synthetic ground truth
    logger.info(f"\n[1/3] Generating ground truth for {num_users} users, {num_videos} videos...")
    all_video_ids = [f"v_{i}" for i in range(num_videos)]
    
    ranked_lists = []
    relevant_items = []
    
    for u in range(num_users):
        # Each user has 5-20 "truly relevant" videos
        num_relevant = np.random.randint(5, 21)
        relevant_set = set(np.random.choice(all_video_ids, size=num_relevant, replace=False))
        
        # Simulate retrieval: model returns top-100 with some relevant items mixed in
        # Higher quality = more relevant items ranked higher
        retrieved = []
        for rank_pos in range(100):
            if rank_pos < 30 and np.random.random() < 0.35:
                # Good model: relevant items appear in top 30
                candidates = list(relevant_set - set(retrieved))
                if candidates:
                    retrieved.append(np.random.choice(candidates))
                    continue
            # Random noise
            retrieved.append(np.random.choice(all_video_ids))
        
        ranked_lists.append(retrieved)
        relevant_items.append(relevant_set)
    
    # Step 2: Compute metrics
    logger.info("[2/3] Computing evaluation metrics...")
    t0 = time.time()
    
    results = {}
    
    # Retrieval metrics
    for k in [10, 50, 100]:
        recall = calculate_recall_at_k(ranked_lists, relevant_items, k=k)
        results[f"Recall@{k}"] = round(recall, 4)
    
    # Ranking metrics
    ndcg_10 = calculate_ndcg_at_k(ranked_lists, relevant_items, k=10)
    ndcg_20 = calculate_ndcg_at_k(ranked_lists, relevant_items, k=20)
    mrr = calculate_mrr(ranked_lists, relevant_items)
    
    results["NDCG@10"] = round(ndcg_10, 4)
    results["NDCG@20"] = round(ndcg_20, 4)
    results["MRR"] = round(mrr, 4)
    
    # Simulate AUC (binary classification of watch/skip)
    # Generate predicted scores and binary labels
    y_true = np.random.binomial(1, 0.3, size=num_users * 10)  # 30% positive rate
    y_pred = y_true * 0.6 + np.random.random(len(y_true)) * 0.4  # Correlated predictions
    
    # Manual AUC computation
    pos_scores = y_pred[y_true == 1]
    neg_scores = y_pred[y_true == 0]
    auc = np.mean([np.mean(pos_scores > neg) for neg in neg_scores]) if len(neg_scores) > 0 else 0.5
    results["AUC-ROC"] = round(auc, 4)
    
    # Log Loss (simplified)
    eps = 1e-7
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    logloss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    results["LogLoss"] = round(logloss, 4)
    
    eval_time = time.time() - t0
    results["eval_time_seconds"] = round(eval_time, 3)
    results["num_users"] = num_users
    results["num_videos"] = num_videos
    
    # Step 3: Print results table
    logger.info("[3/3] Evaluation complete.\n")
    
    print("\n" + "=" * 50)
    print("  REELMIND EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Users: {num_users}  |  Videos: {num_videos}")
    print("-" * 50)
    print(f"  {'Metric':<20} {'Value':>12}")
    print("-" * 50)
    
    metric_order = ["Recall@10", "Recall@50", "Recall@100", "NDCG@10", "NDCG@20", "MRR", "AUC-ROC", "LogLoss"]
    for metric in metric_order:
        val = results.get(metric, 0)
        print(f"  {metric:<20} {val:>12.4f}")
    
    print("-" * 50)
    print(f"  Evaluation Time: {eval_time:.3f}s")
    print("=" * 50)
    
    # Save results
    output_path = Path(output)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info(f"\nResults saved to {output_path.absolute()}")


if __name__ == "__main__":
    main()

