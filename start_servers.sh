#!/bin/bash
# start_servers.sh

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

echo "Waiting 15 seconds for models and services to initialize..."
sleep 15

echo "Populating FAISS Index with mock video data..."
python scripts/populate_index.py

echo ""
echo "================================================================"
echo "✅ All systems go!"
echo "📱 Open your browser and go to: http://localhost:8001"
echo "================================================================"
echo "Press Ctrl+C to stop all services."

# Wait for all background processes, allowing Ctrl+C to kill them
wait $FE_PID $RET_PID $RANK_PID $RL_PID $API_PID
