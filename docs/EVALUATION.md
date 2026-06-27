# Evaluation Metrics & Benchmarking

ReelMind uses industry-standard Information Retrieval (IR) and Classification metrics to evaluate the performance of the ML models offline before they are deployed to the live system.

## Running the Evaluation

To run the offline evaluation pipeline against the current models, use the training CLI:

```bash
PYTHONPATH=. python -m ml.training.cli evaluate --num-users 200 --num-videos 500
```

This generates synthetic ground-truth data, passes it through the local retrieval and ranking models, computes the metrics, and saves a report to `evaluation_results.json`.

---

## Metrics Explained

### 1. Retrieval Metrics: Recall@K
Used primarily to evaluate the **Two-Tower Candidate Generator** and **FAISS index**.

* **What it is:** Out of all the videos a user actually engaged with (the ground truth positive set), what percentage were successfully retrieved in the top K results by the model?
* **Why it matters:** The retrieval stage is a funnel. If a highly relevant video is not caught in the initial top 100 FAISS retrieval, the DeepFM ranker will never even see it. High recall is critical at this stage.
* **Our Targets:**
  * Recall@100: > 0.90
  * Recall@50: > 0.80

### 2. Ranking Metrics: NDCG@K
Normalized Discounted Cumulative Gain (NDCG) is used to evaluate the **DeepFM Heavy Ranker**.

* **What it is:** Measures ranking quality by penalizing relevant items that appear lower in the list. It compares the model's ranked list against an "ideal" ranking where all relevant items are at the very top.
* **Why it matters:** In a short-video feed, users rarely scroll past the first few videos if they are bad. Finding relevant content is not enough; the *most* relevant content must be at Rank 1.
* **Our Targets:**
  * NDCG@10: > 0.35
  * NDCG@20: > 0.45

### 3. Ranking Metrics: MRR
Mean Reciprocal Rank (MRR) evaluates how quickly the model serves a "hit."

* **What it is:** The average of the reciprocal of the rank of the *first* relevant item. If the first relevant item is at rank 1, score = 1.0. If it's at rank 3, score = 0.33.
* **Why it matters:** Captures the "time to first smile." The user needs to see something they like almost immediately upon opening the app.
* **Our Target:** MRR > 0.50 (On average, a highly relevant video appears in the top 2 slots).

### 4. Classification Metrics: AUC-ROC & LogLoss
Used to evaluate the DeepFM model's raw predictive power as a binary classifier (Watch vs. Skip).

* **AUC-ROC:** Area Under the Receiver Operating Characteristic Curve. Represents the probability that the model will rank a randomly chosen positive item (a watched video) higher than a randomly chosen negative item (a skipped video). 0.5 is random guessing, 1.0 is perfect.
* **LogLoss:** Measures the calibration of the predicted probabilities. A lower score is better, indicating the model's confidence aligns closely with reality.

---

## Systems Benchmarking (Latency & Throughput)

Beyond ML quality, recommendation systems are constrained by strict latency budgets. To test the system's ability to handle concurrent traffic, use the benchmark script:

```bash
python scripts/benchmark.py -c 20 -n 200 -u http://localhost:8001
```

This simulates 20 concurrent users requesting 200 total feeds, measuring the end-to-end latency of the API Gateway orchestrating the 5 microservices.

**Typical Latency Budget Breakdown (< 100ms Total):**
1. Feature Fetch (Redis/Mem): < 10ms
2. Two-Tower FAISS Retrieval: < 15ms
3. LightGBM Pre-Rank: < 10ms
4. DeepFM Heavy Rank: < 40ms
5. RL Optimization: < 10ms
6. Network Overhead: < 15ms
