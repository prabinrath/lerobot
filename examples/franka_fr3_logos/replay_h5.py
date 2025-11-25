#!/usr/bin/env python

# ---------------------------------------------------------------------------
# Copyright (c) 2025 Prabin Kumar Rath
# Co-developed with Claude Sonnet 4.5 (Anthropic)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------

"""
Replay H5 demonstrations on Franka FR3 robot.

This script reads joint states from an H5 file and replays them on the robot
at the specified FPS (frames per second).

Usage:
    python replay_h5.py --h5_path path/to/dataset.h5 --fps 10
    
    Optional arguments:
    --demo_index 0  # Replay a specific demo (default: 0)
"""

import argparse
import logging
import time
from pathlib import Path

import h5py
import numpy as np

from lerobot.robots.franka_fr3.config_franka_fr3 import FrankaFR3Config
from lerobot.robots.utils import make_robot_from_config


def extract_joint_positions_from_h5(joint_states):
    """
    Extract joint positions from H5 joint_states data.
    
    Args:
        joint_states: Array of shape (timesteps, 3, N) where N is 8 or 9
                     containing [position, velocity, acceleration]
        
    Returns:
        Array of shape (timesteps, 8) containing 7 joint positions + gripper value
    """
    # Extract position data (first row) - use first 8 joints (7 arm + 1 gripper)
    joint_positions = joint_states[:, 0, :8]  # Shape: (timesteps, 8)
    return joint_positions


def replay_h5_on_robot(
    h5_path: str,
    fps: int = 10,
    demo_index: int = 0,
    robot_id: str = "franka_fr3",
    logger: logging.Logger | None = None,
):
    """
    Replay H5 demonstration on Franka FR3 robot.
    
    Args:
        h5_path: Path to the H5 dataset file
        fps: Playback frames per second
        demo_index: Index of the demo to replay (0-indexed)
        robot_id: Unique identifier for the robot
        logger: Logger instance for logging messages
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Replaying H5 demonstration from {h5_path}")
    logger.info(f"FPS: {fps}")
    logger.info(f"Demo index: {demo_index}")
    
    # Validate H5 file exists
    h5_file = Path(h5_path)
    if not h5_file.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    # Load demonstration from H5 file
    logger.info("Loading demonstration data...")
    with h5py.File(h5_path, 'r') as f:
        data_group = f['data']
        demo_keys = [key for key in data_group.keys() if key.startswith('demo_')]
        demo_keys.sort(key=lambda x: int(x.split('_')[1]))
        
        if demo_index >= len(demo_keys):
            raise ValueError(f"Demo index {demo_index} out of range (0-{len(demo_keys)-1})")
        
        demo_key = demo_keys[demo_index]
        logger.info(f"Loading {demo_key} ({demo_index + 1}/{len(demo_keys)})")
        
        demo_data = data_group[demo_key]
        joint_states = np.array(demo_data['joint_states'])
    
    # Extract joint positions
    joint_positions = extract_joint_positions_from_h5(joint_states)
    num_frames = len(joint_positions)
    logger.info(f"Loaded {num_frames} frames")
    
    # Create minimal robot configuration (no cameras needed for replay)
    robot_config = FrankaFR3Config(
        id=robot_id,
        cameras={},  # No cameras needed for replay
        dt=1.0/fps
    )
    
    # Connect to robot
    logger.info("Connecting to robot...")
    robot = make_robot_from_config(robot_config)
    robot.connect()
    logger.info("Robot connected! Starting replay (Ctrl+C to stop)...")
    
    try:
        dt = 1.0 / fps
        
        for frame_idx in range(num_frames):
            start_time = time.perf_counter()
            
            # Get joint positions for this frame
            target_joints = joint_positions[frame_idx]
            
            action = {}
            for i, joint_name in enumerate(robot_config.joint_names):
                action[f"{joint_name}.pos"] = float(target_joints[i])
            
            # Send action to robot
            robot.send_action(action)
            
            # Log progress periodically
            if frame_idx % fps == 0:
                logger.info(f"Frame {frame_idx + 1}/{num_frames} ({100 * (frame_idx + 1) / num_frames:.1f}%)")
            
            # Maintain control frequency
            elapsed = time.perf_counter() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
        
        logger.info("Replay completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Replay interrupted by user")
    finally:
        robot.disconnect()
        logger.info("Robot disconnected")


def main():
    # Set up logger
    logger = logging.getLogger("replay_h5")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    logger.propagate = False
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Replay H5 demonstrations on Franka FR3 robot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        default="../../datasets/tiger_in_basket.h5",
        help="Path to the H5 dataset file"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Playback frames per second"
    )
    parser.add_argument(
        "--demo_index",
        type=int,
        default=0,
        help="Index of the demo to replay (0-indexed)"
    )
    parser.add_argument(
        "--robot_id",
        type=str,
        default="franka_fr3",
        help="Unique identifier for the robot"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    logger.info("="*70)
    logger.info("Franka FR3 H5 Replay Script")
    logger.info("="*70)
    logger.info(f"H5 File: {args.h5_path}")
    logger.info(f"FPS: {args.fps}")
    logger.info(f"Demo Index: {args.demo_index}")
    logger.info(f"Robot ID: {args.robot_id}")
    logger.info("="*70)
    
    # Replay demonstration
    replay_h5_on_robot(
        h5_path=args.h5_path,
        fps=args.fps,
        demo_index=args.demo_index,
        robot_id=args.robot_id,
        logger=logger,
    )


if __name__ == "__main__":
    main()
