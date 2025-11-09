#!/usr/bin/env python

"""
Franka FR3 Robot Client for Policy Deployment

This script connects to your Franka FR3 robot and communicates with the policy server
to execute your trained policy (ACT, Diffusion, VQ-BeT, SmolVLA, GROOT, PI0, PI05) asynchronously.

Usage:
    python deploy_robot_client.py [OPTIONS]

Example (with RealSense cameras):
    python deploy_robot_client.py \
        --server_address 127.0.0.1:8080 \
        --checkpoint_path outputs/train/act_franka_fr3_softtoy/checkpoints/last \
        --policy_type act \
        --task "pick up the soft toy and place it in the drawer" \
        --cameras "{'front_img': {'serial_number_or_name': '938422074102', 'width': 640, 'height': 480, 'fps': 30, 'rotation': 180}, 'wrist_img': {'serial_number_or_name': '919122070360', 'width': 640, 'height': 480, 'fps': 30}}"
"""

import argparse
import ast
import logging
import threading
from pathlib import Path

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots.franka_fr3.config_franka_fr3 import FrankaFR3Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Start Franka FR3 robot client for policy deployment (supports ACT, Diffusion, VQ-BeT, SmolVLA, GROOT, PI0, PI05)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Server configuration
    parser.add_argument(
        "--server_address", 
        type=str, 
        default="127.0.0.1:8080",
        help="Address of the policy server (host:port)"
    )
    
    # Franka FR3 configuration
    parser.add_argument(
        "--robot_id", 
        type=str, 
        default="franka_fr3",
        help="Unique identifier for the robot (used for calibration files)"
    )
    
    # Policy configuration
    parser.add_argument(
        "--policy_type", 
        type=str, 
        required=True,
        choices=["act", "diffusion", "vqbet", "smolvla", "groot", "pi0", "pi05"],
        help="Type of policy to use"
    )
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=True,
        help="Path to the trained policy checkpoint"
    )
    parser.add_argument(
        "--policy_device", 
        type=str, 
        default="cuda",
        help="Device for policy inference (cuda, cpu, mps)"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default="",
        help="Task description for the robot to execute"
    )
    
    # Camera configuration
    parser.add_argument(
        "--cameras", 
        type=str, 
        required=True,
        help="Camera configuration as a Python dict string where keys are camera names and values are RealSenseCameraConfig parameters. Example: \"{'front_img': {'serial_number_or_name': '938422074102', 'width': 640, 'height': 480, 'fps': 30, 'rotation': 180}, 'wrist_img': {'serial_number_or_name': '919122070360', 'width': 640, 'height': 480, 'fps': 30}}\""
    )
    
    # Control parameters
    parser.add_argument(
        "--actions_per_chunk", 
        type=int, 
        default=16,
        help="Number of actions per inference chunk (ACT: 5-20, VQ-BeT: 10-50, etc.)"
    )
    parser.add_argument(
        "--chunk_size_threshold", 
        type=float, 
        default=0.5,
        help="Threshold for requesting new actions (0.0-1.0, higher = more responsive)"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=10,
        help="Control frequency in frames per second (should match training data frequency)"
    )
    parser.add_argument(
        "--aggregate_fn_name", 
        type=str, 
        default="weighted_average",
        choices=["weighted_average", "latest_only", "average", "conservative"],
        help="Function to aggregate overlapping action chunks"
    )
    
    # Safety parameters
    parser.add_argument(
        "--max_relative_target", 
        type=float, 
        default=0.05,
        help="Maximum relative joint movement per step for safety (robot-specific)"
    )
    
    # Debug options
    parser.add_argument(
        "--debug_visualize_queue_size", 
        action="store_true",
        help="Visualize action queue size during execution (useful for tuning)"
    )
    parser.add_argument(
        "--dry_run", 
        action="store_true",
        help="Test configuration without connecting to robot"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    if not 0.0 <= args.chunk_size_threshold <= 1.0:
        raise ValueError(f"chunk_size_threshold must be between 0.0 and 1.0, got {args.chunk_size_threshold}")
    
    if args.actions_per_chunk <= 0:
        raise ValueError(f"actions_per_chunk must be positive, got {args.actions_per_chunk}")
    
    # Parse camera configuration
    try:
        cameras_dict = ast.literal_eval(args.cameras)
        
        # Map rotation degrees to Cv2Rotation enum
        rotation_map = {
            0: Cv2Rotation.NO_ROTATION,
            90: Cv2Rotation.ROTATE_90,
            180: Cv2Rotation.ROTATE_180,
            270: Cv2Rotation.ROTATE_270,
        }
        
        camera_configs = {}
        for camera_name, config in cameras_dict.items():
            # Convert rotation if present
            if 'rotation' in config:
                config = {**config, 'rotation': rotation_map[config['rotation']]}
            
            camera_configs[camera_name] = RealSenseCameraConfig(**config)
            
    except Exception as e:
        logger.error(f"Failed to parse camera configuration: {e}")
        return 1
    
    # Create Franka FR3 robot configuration
    robot_config = FrankaFR3Config(
        id=args.robot_id,
        cameras=camera_configs,
    )
    
    # Add safety parameters if provided
    if args.max_relative_target is not None:
        robot_config.max_relative_target = args.max_relative_target
    
    # Create client configuration
    client_config = RobotClientConfig(
        robot=robot_config,
        server_address=args.server_address,
        policy_type=args.policy_type,
        pretrained_name_or_path=str(checkpoint_path),
        policy_device=args.policy_device,
        task=args.task,
        actions_per_chunk=args.actions_per_chunk,
        chunk_size_threshold=args.chunk_size_threshold,
        fps=args.fps,
        aggregate_fn_name=args.aggregate_fn_name,
        debug_visualize_queue_size=args.debug_visualize_queue_size,
    )
    
    # Print configuration summary
    logger.info("="*70)
    logger.info("Franka FR3 Policy Deployment Client")
    logger.info("="*70)
    logger.info(f"Server Address: {client_config.server_address}")
    logger.info(f"Robot ID: {robot_config.id}")
    logger.info(f"Policy Type: {client_config.policy_type.upper()}")
    logger.info(f"Policy Checkpoint: {checkpoint_path}")
    logger.info(f"Policy Device: {client_config.policy_device}")
    logger.info(f"Task: {client_config.task or 'No task specified'}")
    logger.info(f"Cameras ({len(camera_configs)}):")
    for name, cfg in camera_configs.items():
        logger.info(f"  - {name}: {cfg.serial_number_or_name} @ {cfg.width}x{cfg.height} {cfg.fps}fps" + 
                   (f" (rotation: {cfg.rotation.value}°)" if hasattr(cfg, 'rotation') and cfg.rotation.value != 0 else ""))
    logger.info(f"Actions per Chunk: {client_config.actions_per_chunk}")
    logger.info(f"Chunk Size Threshold: {client_config.chunk_size_threshold}")
    logger.info(f"FPS: {client_config.fps}")
    logger.info(f"Max Relative Target: {getattr(robot_config, 'max_relative_target', 'Not set')}")
    logger.info(f"Aggregate Function: {client_config.aggregate_fn_name}")
    logger.info("="*70)
    
    if args.dry_run:
        logger.info("DRY RUN MODE: Configuration validated successfully!")
        logger.info("Remove --dry_run flag to actually connect to the robot.")
        return 0
    
    # Create and start robot client
    client = RobotClient(client_config)
    
    try:
        logger.info("Connecting to policy server and robot...")
        if not client.start():
            logger.error("Failed to start robot client")
            return 1
        
        logger.info("Successfully connected! Starting control loop...")
        logger.info("Press Ctrl+C to stop execution.")
        
        # Start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()
        
        # Run the control loop
        client.control_loop(args.task)
        
    except KeyboardInterrupt:
        logger.info("Stopping robot client...")
        client.stop()
        action_receiver_thread.join(timeout=5.0)
        
        # Optionally visualize queue size statistics
        if args.debug_visualize_queue_size and hasattr(client, 'action_queue_size'):
            logger.info("Displaying action queue size visualization...")
            visualize_action_queue_size(client.action_queue_size)
        
        logger.info("Robot client stopped successfully.")
        
    except Exception as e:
        logger.error(f"Robot client error: {e}")
        client.stop()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
