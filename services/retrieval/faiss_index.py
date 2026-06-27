"""
FAISS Approximate Nearest Neighbor (ANN) Index.

Manages the indexing of millions of video embeddings for sub-millisecond retrieval.
Uses Inverted File (IVF) with Product Quantization (PQ) for memory efficiency
and high query throughput.
"""

import logging
import os
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndex:
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.index: Optional[faiss.Index] = None
        self.video_ids: list[str] = []
        
    def build_index(
        self, 
        video_ids: list[str], 
        embeddings: np.ndarray,
        nlist: int = 100,  # Number of Voronoi cells (clusters)
        nprobe: int = 10,  # Number of clusters to search
        use_pq: bool = False
    ) -> None:
        """
        Build the FAISS index from video embeddings.
        
        Args:
            video_ids: List of video IDs corresponding to the embeddings
            embeddings: Float32 numpy array of shape (N, dim)
            nlist: Number of clusters for IVF
            nprobe: Number of clusters to search during query
            use_pq: Whether to use Product Quantization for compression
        """
        assert len(video_ids) == len(embeddings), "Video IDs and embeddings count mismatch"
        assert embeddings.shape[1] == self.embedding_dim, f"Expected {self.embedding_dim}d embeddings"
        
        self.video_ids = video_ids
        num_items = len(embeddings)
        
        # Adjust nlist if we have very few items (e.g. in tests)
        nlist = min(nlist, max(1, num_items // 39))
        
        logger.info(f"Building FAISS index for {num_items} items (nlist={nlist})...")
        
        # Base quantizer (L2 distance is equivalent to cosine similarity for normalized vectors)
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        
        if use_pq:
            # IVFPQ: Inverted File with Product Quantization
            # m=8 subquantizers, 8 bits each
            self.index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, 8, 8)
        else:
            # IVFFlat: Inverted File without compression (exact distances)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            
        # Train the index (clusters the data)
        self.index.train(embeddings)
        
        # Add the vectors
        self.index.add(embeddings)
        
        # Set search parameters
        self.index.nprobe = min(nprobe, nlist)
        
        logger.info("FAISS index build complete.")
        
    def search(self, query_embedding: np.ndarray, top_k: int = 100) -> list[tuple[str, float]]:
        """
        Search for the top-K most similar videos to the query embedding.
        
        Args:
            query_embedding: Numpy array of shape (1, dim)
            top_k: Number of results to return
            
        Returns:
            List of (video_id, similarity_score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
            
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        assert query_embedding.shape[1] == self.embedding_dim
        
        # Search returns distances and indices
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            if idx != -1:  # -1 means not enough results found
                results.append((self.video_ids[idx], float(dist)))
                
        return results
        
    def save(self, filepath: str) -> None:
        """Save the index and video IDs to disk."""
        if self.index is None:
            raise ValueError("Index not built.")
            
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save mapping
        with open(f"{filepath}.mapping", "w") as f:
            for vid in self.video_ids:
                f.write(f"{vid}\n")
                
        logger.info(f"Saved FAISS index to {filepath}")
        
    def load(self, filepath: str) -> None:
        """Load the index and video IDs from disk."""
        if not os.path.exists(f"{filepath}.index"):
            raise FileNotFoundError(f"Index file not found: {filepath}.index")
            
        self.index = faiss.read_index(f"{filepath}.index")
        
        self.video_ids = []
        with open(f"{filepath}.mapping", "r") as f:
            for line in f:
                self.video_ids.append(line.strip())
                
        logger.info(f"Loaded FAISS index with {len(self.video_ids)} items.")
