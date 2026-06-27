"""
Load testing script for ReelMind recommendation pipeline.
"""

import asyncio
import time
import httpx
import logging
import argparse
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_feed(client: httpx.AsyncClient, base_url: str, user_id: str) -> float:
    start_time = time.time()
    try:
        response = await client.post(
            f"{base_url}/v1/feed",
            json={"user_id": user_id, "num_results": 15}
        )
        response.raise_for_status()
        return (time.time() - start_time) * 1000
    except Exception as e:
        logger.error(f"Error fetching feed for {user_id}: {e}")
        return -1.0

async def run_benchmark(concurrency: int, total_requests: int, base_url: str):
    logger.info(f"Starting benchmark: {total_requests} requests, concurrency {concurrency}")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        latencies: List[float] = []
        
        # Create batches of requests
        batches = [total_requests // concurrency] * concurrency
        remainder = total_requests % concurrency
        for i in range(remainder):
            batches[i] += 1
            
        start_time = time.time()
        
        async def worker(num_reqs: int, worker_id: int):
            for i in range(num_reqs):
                user_id = f"u_{worker_id * 1000 + i}"
                lat = await fetch_feed(client, base_url, user_id)
                if lat > 0:
                    latencies.append(lat)
                    
        tasks = [worker(num, i) for i, num in enumerate(batches)]
        await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        if not latencies:
            logger.error("All requests failed!")
            return
            
        latencies.sort()
        successful = len(latencies)
        qps = successful / total_time
        
        logger.info("=== Benchmark Results ===")
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info(f"Successful Requests: {successful}/{total_requests}")
        logger.info(f"Throughput: {qps:.2f} QPS")
        logger.info("Latencies (ms):")
        logger.info(f"  Min: {latencies[0]:.2f}")
        logger.info(f"  P50: {latencies[int(successful * 0.50)]:.2f}")
        logger.info(f"  P90: {latencies[int(successful * 0.90)]:.2f}")
        logger.info(f"  P95: {latencies[int(successful * 0.95)]:.2f}")
        logger.info(f"  P99: {latencies[int(successful * 0.99)]:.2f}")
        logger.info(f"  Max: {latencies[-1]:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--concurrency", type=int, default=10)
    parser.add_argument("-n", "--requests", type=int, default=100)
    parser.add_argument("-u", "--url", type=str, default="http://localhost:8001")
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(args.concurrency, args.requests, args.url))
