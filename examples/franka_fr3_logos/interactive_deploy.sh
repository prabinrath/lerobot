#!/bin/bash

# Interactive Deploy - Launches Policy Server, Robot Client (Interactive), and Interaction Server
# Kill all processes with Ctrl+C

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Store PIDs
PIDS=()

# Trap Ctrl+C and kill all processes
cleanup() {
    echo "Stopping all processes..."
    # First try graceful shutdown with SIGTERM
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null
    done
    # Wait briefly for graceful shutdown
    sleep 0.5
    # Force kill any remaining processes
    for pid in "${PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null
    done
    exit
}

trap cleanup SIGINT SIGTERM

# Start Policy Server
python "$SCRIPT_DIR/deploy_policy_server.py" \
    --host 127.0.0.1 \
    --port 8080 \
    --fps 10 \
    --similarity_atol 0.15 &
PIDS+=($!)

# Start Robot Client (Interactive)
python "$SCRIPT_DIR/deploy_robot_client.py" \
    --interactive \
    --server_address 127.0.0.1:8080 \
    --robot_id franka_fr3 \
    --chunk_size_threshold 0.5 \
    --actions_per_chunk 50 \
    --fps 10 \
    --max_rollout_steps 800 &
PIDS+=($!)

# Start Interaction Server
python "$SCRIPT_DIR/interaction_server.py" &
PIDS+=($!)

# Wait for all background processes
wait
