#!/bin/bash
# run_local.sh

echo "Starting ReelMind Microservices Locally..."

export PYTHONPATH=.
export KMP_DUPLICATE_LIB_OK=TRUE

# Start the services in the background
python -m uvicorn services.feature_engine.main:app --port 8002 &
FE_PID=$!

python -m uvicorn services.retrieval.main:app --port 8003 &
RET_PID=$!

python -m uvicorn services.ranking.main:app --port 8004 &
RANK_PID=$!

python -m uvicorn services.rl_optimizer.main:app --port 8005 &
RL_PID=$!

python -m uvicorn services.api_gateway.main:app --port 8001 &
API_PID=$!

# Give them a few seconds to start up
echo "Waiting for services to initialize..."
sleep 15

echo "Populating FAISS Index..."
python scripts/populate_index.py

echo "Running Benchmark Script..."
python scripts/benchmark.py --concurrency 5 --requests 20

echo "Benchmark complete. Shutting down services..."
kill $FE_PID $RET_PID $RANK_PID $RL_PID $API_PID
echo "Done!"
