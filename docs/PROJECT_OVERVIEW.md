# ReelMind Project Overview

This document provides a complete, top-to-bottom breakdown of the entire project. Use this as a guide for presentations, interviews, or understanding the full scope of the system.

---

## 1. What Happens in the Backend? (How it Works)
The backend is built using a **microservice architecture**. Instead of one giant application, it is broken into 5 separate Python/FastAPI services that talk to each other:

1. **Feature Engine (Port 8002):** Fetches the data. When a user requests a feed, this service grabs the user's historical data and the metadata of the videos.
2. **Retrieval Service (Port 8003):** Runs a **Two-Tower PyTorch model** to encode the user's data into a mathematical vector. It searches a **FAISS vector database** (which holds 1,000 synthetic videos) to find the 100 most similar videos in under 5 milliseconds.
3. **Ranking Service (Port 8004):** Takes those 100 videos and runs them through two heavy Machine Learning models:
   * **LightGBM (Pre-ranker):** A fast tree model that filters the 100 down to 50.
   * **DeepFM (Heavy ranker):** A deep neural network that gives a precise "Watch Probability Score" (0% to 100%) to the final 50 videos.
4. **RL Optimizer (Port 8005):** A Reinforcement Learning model using **Thompson Sampling**. Instead of just showing the highest-scored videos (which creates a boring "filter bubble"), it injects diversity (e.g., throwing a sports video into a gaming feed to see if you like it).
5. **API Gateway (Port 8001):** The "Boss". It orchestrates all the other 4 services and acts as the single point of contact for the frontend website.

---

## 2. What Pages are on the Website & What Do They Show?
The frontend is a lightweight HTML/JS/CSS website running on `http://localhost:8001`. It has **three main pages**, accessible via a top navigation bar:

### Page 1: The Main Feed (`/`)
* **What it is:** A TikTok-style scrolling interface. 
* **Features:**
  * **Infinite Scroll & Autoplay:** Videos play automatically as you scroll to them.
  * **User Switcher:** A dropdown at the top to switch between simulated users.
  * **Cold Start Button:** A `+ New User` button that generates a brand new user with absolutely zero history to test the AI's exploration capability.
  * **Interaction Buttons:** Working "Like", "Skip", "Share", and "Comment" buttons.
* **What's actually happening:** When you click "Like" or scroll past a video quickly, the website sends an Event back to the API Gateway. The backend instantly learns what you like and recalculates your next feed.

### Page 2: ML Analytics Dashboard (`/analytics`)
* **What it is:** The "Proof" page showing what the AI is thinking about the user in real-time. It auto-refreshes every 3 seconds.
* **What it displays:**
  * **Session Stats:** Total watch time, number of skips, and your "Retention Score".
  * **Category Affinities:** Progress bars showing exactly how much the system thinks you like "Comedy", "Tech", "Sports", etc. (If you like 3 comedy videos on the main feed, this bar instantly shoots up).
  * **Pipeline Latency:** A horizontal bar chart showing exactly how many milliseconds the feature engine, retrieval, ranking, and RL models took to load the last feed.
  * **Feed Explanation:** A table showing the exact videos currently in the user's feed, including their deep learning Score, Rank, and whether the RL model placed them there for "Exploration".

### Page 3: System Pipeline Dashboard (`/dashboard`)
* **What it is:** The "Infrastructure" observability page.
* **What it displays:** A massive stacked bar chart. When you click **"Run Simulation"**, it simulates 15 rapid-fire user requests. It graphs the exact millisecond latency breakdown across all 5 microservices, proving that the system is distributed and runs under 100ms.

---

## 3. How Does it Train?
The system has two types of learning: **Offline Training** and **Online Learning**.

### 1. Offline Training (The Neural Networks)
There is a command-line interface (`ml/training/cli.py`). You can run a command that simulates a dataset of thousands of users and videos. It trains the PyTorch Two-Tower and DeepFM models, calculates their accuracy using standard formulas (`NDCG@10`, `MRR`, `Recall@100`), and saves the model weights to disk. 

### 2. Online Learning (The Live Feedback Loop)
The deep neural networks take hours to train, so they are fixed while the app runs. **BUT**, the RL Optimizer (The DJ) learns *instantly*. 
When you click "Like" on the website, the API Gateway immediately updates the mathematical probability (the "Posterior Distribution") of that specific category in memory. So, if you refresh the website, your feed adapts immediately without needing to retrain the heavy neural networks.

---

## Summary
ReelMind is an extremely realistic simulation of how big tech companies rank content. It utilizes a TikTok-style UI that sends real interaction events to a 5-stage microservice architecture, which runs real deep learning and reinforcement learning algorithms to calculate the perfect feed in under 100 milliseconds.
