"""
Train the DeepFM Heavy Ranker.
"""

import logging
import os
import time

import mlflow
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from services.ranking.deepfm import DeepFMModel
from ml.training.dataset import ReelMindDataset

logger = logging.getLogger(__name__)


def train_deepfm(data_dir: str, epochs: int = 5, batch_size: int = 512, lr: float = 0.001):
    logger.info(f"Starting DeepFM training on data from {data_dir}")
    
    mlflow.set_experiment("deepfm_ranking")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "model_type": "deepfm"
        })
        
        try:
            dataset = ReelMindDataset(data_dir)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return
            
        model = DeepFMModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()  # Binary cross entropy for CTR
        
        start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            
            for batch_idx, (u_feat, v_feat, c_feat, label) in enumerate(dataloader):
                # Combine features
                x = torch.cat([u_feat, v_feat, c_feat], dim=1).to(device)
                y_true = label.to(device)
                
                optimizer.zero_grad()
                
                y_pred = model(x)
                loss = criterion(y_pred, y_true)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs} - BCE Loss: {avg_loss:.4f}")
            mlflow.log_metric("bce_loss", avg_loss, step=epoch)
            
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.1f}s")
        
        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/deepfm_latest.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        mlflow.log_artifact(model_path)
        
    return model_path
