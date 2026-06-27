"""
Train the Two-Tower Retrieval Model.
"""

import logging
import os
import time

import mlflow
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from services.retrieval.two_tower import TwoTowerModel
from ml.training.dataset import ReelMindDataset

logger = logging.getLogger(__name__)


def train_two_tower(data_dir: str, epochs: int = 5, batch_size: int = 256, lr: float = 0.001):
    logger.info(f"Starting Two-Tower training on data from {data_dir}")
    
    # MLflow tracking
    mlflow.set_experiment("two_tower_retrieval")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "model_type": "two_tower"
        })
        
        # Load dataset
        try:
            dataset = ReelMindDataset(data_dir)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            logger.info(f"Loaded {len(dataset)} training examples")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return
            
        model = TwoTowerModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            
            for batch_idx, (u_feat, v_feat, _, label) in enumerate(dataloader):
                u_feat, v_feat = u_feat.to(device), v_feat.to(device)
                
                # In-batch negative sampling logic
                # For positive pairs (where label == 1), we want to maximize similarity.
                # For negative pairs, minimize.
                # A common approach is treating all other items in the batch as negatives
                # (Softmax cross-entropy over the batch).
                
                optimizer.zero_grad()
                
                # Filter to only positives for the anchor pairs
                pos_mask = (label.squeeze() > 0.5)
                
                if pos_mask.sum() < 2:
                    continue  # Skip batch if not enough positives
                    
                u_pos = u_feat[pos_mask]
                v_pos = v_feat[pos_mask]
                
                # Logits: pairwise similarities (batch_pos x batch_pos)
                logits = model(u_pos, v_pos)
                
                # Targets: identity matrix (diagonal elements are the true pairs)
                targets = torch.arange(len(u_pos), device=device)
                
                # Cross-entropy loss maximizes the diagonal (positive pair) 
                # and minimizes off-diagonals (in-batch negatives)
                loss = F.cross_entropy(logits, targets)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.1f}s")
        
        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/two_tower_latest.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        mlflow.log_artifact(model_path)
        
    return model_path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_two_tower("./data/small")
