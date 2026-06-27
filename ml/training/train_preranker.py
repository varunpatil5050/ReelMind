"""
Train the LightGBM Pre-Ranker.
"""

import logging
import os
import time

import lightgbm as lgb
import mlflow
import numpy as np

from ml.training.dataset import ReelMindDataset

logger = logging.getLogger(__name__)


def train_preranker(data_dir: str, num_boost_round: int = 100):
    logger.info(f"Starting LightGBM Pre-Ranker training on data from {data_dir}")
    
    mlflow.set_experiment("lightgbm_preranker")
    
    with mlflow.start_run():
        mlflow.log_params({
            "num_boost_round": num_boost_round,
            "model_type": "lightgbm",
            "objective": "regression",
            "metric": "rmse"
        })
        
        try:
            dataset = ReelMindDataset(data_dir)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return
            
        # Extract features and labels into numpy arrays for LightGBM
        # For a dataset that doesn't fit in memory, we would use chunking
        # but here we load all for simplicity
        
        logger.info("Extracting features for LightGBM...")
        X, y = [], []
        for i in range(len(dataset)):
            u, v, c, l = dataset[i]
            # Combine to 38d vector
            x = np.concatenate([u.numpy(), v.numpy(), c.numpy()])
            X.append(x)
            y.append(l.item())
            
        X = np.array(X)
        y = np.array(y)
        
        # Train/val split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1,
            "seed": 42
        }
        
        start_time = time.time()
        
        logger.info("Training LightGBM model...")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=10)]
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.1f}s")
        
        # Log best iteration metrics
        if model.best_iteration > 0:
            mlflow.log_metric("best_iteration", model.best_iteration)
            
        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/preranker_latest.txt"
        model.save_model(model_path, num_iteration=model.best_iteration)
        logger.info(f"Model saved to {model_path}")
        
        mlflow.log_artifact(model_path)
        
    return model_path
