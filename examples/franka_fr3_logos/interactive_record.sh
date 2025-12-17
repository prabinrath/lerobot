#!/bin/bash

# Interactive Record - Launches Record Dataset and Interaction Server
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

# Start Record Dataset
python "$SCRIPT_DIR/record_dataset.py" \
    --robot_id franka_fr3 \
    --output_path datasets/test_dataset_one.h5 \
    --num_episodes 100 \
    --fps 30 \
    --image_size 224 224 &
PIDS+=($!)

# Start Interaction Server
python "$SCRIPT_DIR/interaction_server.py" &
PIDS+=($!)

# Wait for all background processes
wait
