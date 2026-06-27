# System Architecture & ML Models

This document outlines the machine learning architecture and engineering design of the ReelMind recommendation system. The system implements a modern, multi-stage recommendation pipeline commonly found in large-scale short-video platforms (like TikTok, Instagram Reels, and YouTube Shorts).

## High-Level Pipeline

The recommendation flow is orchestrated by the API Gateway and consists of five distinct stages, executing in under 100ms:

1. **Feature Extraction:** Real-time retrieval of user/item vectors and context.
2. **Candidate Generation (Retrieval):** Fast filtering of millions of items down to ~100-500 candidates.
3. **Pre-Ranking (Stage 1 Ranking):** Fast scoring to filter down to ~50 candidates.
4. **Heavy Ranking (Stage 2 Ranking):** Complex, high-fidelity scoring of the top candidates.
5. **Re-ranking (RL Optimization):** Diversification and exploration injections before final serving.

---

## 1. Candidate Generation (Retrieval)

The retrieval stage is responsible for narrowing down the entire video corpus into a manageable subset. It must optimize for **latency** and **high recall**.

**Model: Two-Tower Neural Network (PyTorch)**
* **Architecture:** Two separate Multi-Layer Perceptrons (MLPs). One encodes the User features into an embedding vector, the other encodes the Video features into an embedding vector of the exact same dimensionality.
* **Loss Function:** Trained using In-Batch Negative Sampling with a Softmax Cross-Entropy loss. The model learns to maximize the dot product (cosine similarity) between a user and videos they engaged with, while minimizing the dot product with other videos in the batch.
* **Serving:** At inference time, only the User Tower is run. The Video Tower embeddings are pre-computed offline.

**Vector Database: FAISS**
* The pre-computed video embeddings are loaded into an in-memory **FAISS (Facebook AI Similarity Search)** index.
* We use an `IndexHNSWFlat` (Hierarchical Navigable Small World) index for Approximate Nearest Neighbor (ANN) search, allowing sub-millisecond retrieval of the top 100 candidates based on the real-time user embedding.

---

## 2. Pre-Ranking (Stage 1)

With ~100 candidates retrieved, we need to apply more specific user-item interactions. However, running deep neural networks on hundreds of items is too slow.

**Model: LightGBM (Gradient Boosting)**
* **Why:** Tree-based models are extremely fast at inference and handle tabular/dense feature sets exceptionally well.
* **Features:** Uses early crossing of user features, video features, and simple historical statistics.
* **Goal:** Quickly scores the 100 candidates and takes the top 50 to pass to the heavy ranker.

---

## 3. Heavy Ranking (Stage 2)

This is the core ML model that optimizes for the actual business objective (e.g., predicting the probability of a "Like" or estimating "Watch Time Percentage").

**Model: DeepFM (Deep Factorization Machine)**
* **Architecture:** Combines a Factorization Machine (FM) with a Deep Neural Network (DNN).
  * *FM Component:* Explicitly models 2nd-order feature interactions (e.g., User Age × Video Category).
  * *DNN Component:* Implicitly models high-order, non-linear feature interactions.
* **Input:** Receives the top 50 candidates. Features include sparse IDs, dense embeddings, contextual features (time of day), and cross-features.
* **Output:** A predicted engagement score (0.0 to 1.0) for each video. The candidates are sorted by this score.

---

## 4. Reinforcement Learning (RL) Optimizer

A pure DeepFM ranker will often create a "filter bubble," showing the user the exact same type of content repeatedly because it has the highest predicted CTR. The RL Optimizer solves this.

**Algorithm: Contextual Bandits (Thompson Sampling)**
* **Goal:** Balance *Exploitation* (showing what we know the user likes) with *Exploration* (showing new categories to discover hidden interests).
* **Mechanism:** 
  * Each category (comedy, tech, sports) is modeled as a "slot machine" arm with a Beta distribution `Beta(α, β)`.
  * When a user likes/watches a video, `α` (successes) increases. When they skip, `β` (failures) increases.
  * Instead of picking the category with the highest average, we sample from the distribution. Categories with high uncertainty get explored occasionally.

**Diversity: Maximal Marginal Relevance (MMR)**
* MMR penalizes items that are too similar to items already placed in the final feed. If the top 3 ranked videos are all "Gaming," MMR reduces the score of the 4th "Gaming" video to force a "Music" or "Comedy" video into the feed, ensuring a diverse scroll experience.

---

## 5. Feature Store & Streaming Feedback

The system employs an online-offline feature architecture.

* **In-Memory Session Store:** Acts as an ultra-low latency cache for the user's current session. As the user swipes and likes videos on the frontend, these events (`/v1/events`) hit the gateway and update the session store in real-time.
* **Live Posteriors:** The RL bandit's Beta distributions are updated on-the-fly. This allows the system to exhibit **online learning** — the feed adapts within a single session without requiring the deep neural networks to be retrained.
