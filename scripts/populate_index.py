"""
Populate the FAISS index with synthetic video embeddings.

Since we haven't trained the model yet, this uses the untrained Two-Tower model
to generate random 128d embeddings for a set of mock videos. We then push these
to the Retrieval service's /v1/index/rebuild endpoint.
"""

import asyncio
import logging
import numpy as np
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RETRIEVAL_URL = "http://localhost:8003/v1/index/rebuild"
NUM_VIDEOS = 1000

async def populate_index():
    logger.info(f"Generating synthetic features for {NUM_VIDEOS} videos...")
    
    # We use the same deterministic random strategy as the Feature Engine
    # to ensure consistency.
    video_ids = []
    video_features = []
    
    for i in range(NUM_VIDEOS):
        vid = f"v_{i}"
        video_ids.append(vid)
        
        # 17-dimensional synthetic feature vector
        rng = np.random.default_rng(i)
        feat = rng.random(17).astype(np.float32).tolist()
        video_features.append(feat)
        
    payload = {
        "video_ids": video_ids,
        "video_features": video_features
    }
    
    logger.info("Sending features to Retrieval Service to rebuild FAISS index...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(RETRIEVAL_URL, json=payload)
            response.raise_for_status()
            logger.info(f"Success! Response: {response.json()}")
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")

if __name__ == "__main__":
    asyncio.run(populate_index())
