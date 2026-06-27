# High-Level Design (HLD) & System Architecture

This document outlines the infrastructure, microservices, and data flow of the ReelMind recommendation platform. While `ARCHITECTURE.md` focuses on the Machine Learning mathematics, this document focuses on **Systems Engineering and Scalability**.

## 1. System Architecture Diagram

```mermaid
graph TD
    %% Frontend
    Client[Client (Web/Mobile)]

    %% API Gateway Layer
    subgraph "API Gateway Layer"
        GW[FastAPI Gateway\n(Port 8001)]
        SS[(Session Store\nIn-Memory Cache)]
    end

    %% Microservices Layer
    subgraph "ML Microservices"
        FE[Feature Engine\n(Port 8002)]
        RET[Retrieval Service\n(Port 8003)]
        RANK[Ranking Service\n(Port 8004)]
        RL[RL Optimizer\n(Port 8005)]
    end

    %% Data Layer
    subgraph "Data Storage"
        FAISS[(FAISS Index)]
        REDIS[(Redis Cache)]
        KAFKA[(Kafka Event Stream)]
        PG[(PostgreSQL Metadata)]
    end

    %% Connections
    Client <-->|HTTP REST| GW
    GW <--> SS
    
    GW --> FE
    FE --> REDIS
    FE --> PG
    
    GW --> RET
    RET --> FAISS
    
    GW --> RANK
    GW --> RL
    
    %% Event Stream
    GW -.->|Async Events| KAFKA
    KAFKA -.->|Stream Processing| FE
    KAFKA -.->|Model Retraining| RANK
```

---

## 2. Component Details

### 2.1 API Gateway (`services/api_gateway/`)
* **Role:** The orchestrator. It receives the initial request, calls the feature engine, passes features to retrieval, passes candidates to ranking, passes ranked items to RL, and returns the final feed.
* **State Management:** Holds the `SessionStore` singleton to track in-session interactions for real-time personalization without hitting the database.

### 2.2 Feature Engine (`services/feature_engine/`)
* **Role:** High-throughput data fetching.
* **Design:** Designed to read from ultra-low latency stores (Redis). It reconstructs the user's historical embedding and fetches video metadata required by the DeepFM model.

### 2.3 Retrieval Service (`services/retrieval/`)
* **Role:** Vector similarity search.
* **Design:** Wraps a PyTorch Two-Tower user encoder and a FAISS index. FAISS runs completely in RAM. The service exposes a `/v1/index/rebuild` endpoint to hot-swap the vector index when new videos are embedded by the offline pipeline.

### 2.4 Ranking Service (`services/ranking/`)
* **Role:** Compute-heavy ML inference.
* **Design:** Wraps the LightGBM and DeepFM models. DeepFM performs matrix multiplications (CPU/GPU bound). In a production environment, this service scales horizontally (more pods) and utilizes GPU inference endpoints (e.g., NVIDIA Triton) to handle the heavy mathematical load.

### 2.5 RL Optimizer (`services/rl_optimizer/`)
* **Role:** Online learning and diversity.
* **Design:** Lightweight probability service. Holds the Thompson Sampling Beta distributions in memory. Extremely fast execution (< 2ms) to re-order the array of 50 candidates returned by the Ranking Service.

---

## 3. Data Flow

### The Read Path (Recommendation Request)
1. Frontend calls `POST /v1/feed` with `user_id`.
2. Gateway calls Feature Engine to get User Features.
3. Gateway calls Retrieval Service with User Features. Returns Top 100 Video IDs.
4. Gateway calls Feature Engine to get Video Features for those 100 IDs.
5. Gateway calls Ranking Service with (User Features, 100 Video Features). Returns Top 50 scored and sorted videos.
6. Gateway calls RL Optimizer with Top 50 videos and User's Session Profile. Returns Top 15 diverse videos.
7. Gateway responds to Frontend.

### The Write Path (Event Telemetry)
1. Frontend captures a "Like" or "Skip" or "Watch > 50%".
2. Frontend calls `POST /v1/events` asynchronously.
3. Gateway intercepts the event and updates the `SessionStore`.
4. Gateway sends a non-blocking API call to the RL Optimizer to update the Thompson Sampling Beta distributions (`alpha` or `beta` increments).
5. (Production) Event is dropped into a Kafka topic for offline batch processing and nightly model retraining.

---

## 4. Scalability & Fault Tolerance Considerations

* **Stateless Microservices:** With the exception of the API Gateway's session store (which should be moved to Redis in a multi-node deployment) and the RL Optimizer's distributions, all ML inference nodes are completely stateless and can be scaled horizontally behind a load balancer.
* **Fallback Mechanisms:** If the Ranking service times out (takes > 50ms), the API Gateway can bypass it and return the raw output from the Retrieval service. This guarantees the user always gets a feed, even if it's slightly less personalized.
* **Hot-Swapping Vectors:** The FAISS index must be updated without downtime. We do this via an atomic pointer swap in memory when the `/v1/index/rebuild` endpoint is called.
