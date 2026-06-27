"""
Async HTTP clients for internal microservice communication.
"""

import httpx
import logging
from typing import Any

logger = logging.getLogger(__name__)

class ServiceClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=5.0)
        
    async def get(self, endpoint: str, **kwargs) -> Any:
        try:
            response = await self.client.get(endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"GET {self.base_url}{endpoint} failed: {e}")
            raise
            
    async def post(self, endpoint: str, json: dict, **kwargs) -> Any:
        try:
            response = await self.client.post(endpoint, json=json, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"POST {self.base_url}{endpoint} failed: {e}")
            raise

    async def close(self):
        await self.client.aclose()


class FeatureEngineClient(ServiceClient):
    def __init__(self, url: str = "http://localhost:8002"):
        super().__init__(url)
        
    async def get_user_features(self, user_id: str) -> dict:
        return await self.post("/v1/features/user", json={"user_id": user_id})
        
    async def get_video_features_batch(self, video_ids: list[str]) -> dict:
        return await self.post("/v1/features/video/batch", json={"video_ids": video_ids})
        
    async def get_cross_features_batch(self, user_id: str, video_ids: list[str]) -> dict:
        return await self.post("/v1/features/cross/batch", json={"user_id": user_id, "video_ids": video_ids})


class RetrievalClient(ServiceClient):
    def __init__(self, url: str = "http://localhost:8003"):
        super().__init__(url)
        
    async def get_candidates(self, user_id: str, user_vector: list[float], num_candidates: int = 100) -> dict:
        payload = {
            "user_id": user_id,
            "user_features": user_vector,
            "num_candidates": num_candidates,
            "exclude_video_ids": []
        }
        return await self.post("/v1/retrieve", json=payload)


class RankingClient(ServiceClient):
    def __init__(self, url: str = "http://localhost:8004"):
        super().__init__(url)
        
    async def prerank(self, payload: dict) -> dict:
        return await self.post("/v1/prerank", json=payload)
        
    async def rank(self, payload: dict) -> dict:
        return await self.post("/v1/rank", json=payload)


class RLOptimizerClient(ServiceClient):
    def __init__(self, url: str = "http://localhost:8005"):
        super().__init__(url)
        
    async def optimize(self, payload: dict) -> dict:
        return await self.post("/v1/rerank", json=payload)
