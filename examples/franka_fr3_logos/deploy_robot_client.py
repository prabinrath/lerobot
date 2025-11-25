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
Franka FR3 Robot Client for Policy Deployment

This script connects to your Franka FR3 robot and executes your trained policy 
(ACT, Diffusion, VQ-BeT, SmolVLA, GROOT, PI0, PI05) either synchronously (locally) 
or asynchronously (via policy server).

Inference Modes:
    - Async (default): Policy runs on a remote server
    - Sync (--use_sync_inference): Policy runs locally
Usage:
    python deploy_robot_client.py [OPTIONS]

Example (async):
    python deploy_robot_client.py \
        --server_address 127.0.0.1:8080 \
        --checkpoint_path outputs/train/act_franka_fr3_softtoy/checkpoints/last/pretrained_model \
        --policy_type act \
        --task "pick up the soft toy and place it in the drawer"
        # --rename_map '{"observation.images.front_img": "observation.images.camera1", "observation.images.wrist_img": "observation.images.camera2"}'

Example (sync):
    python deploy_robot_client.py \
        --use_sync_inference \
        --checkpoint_path outputs/train/diffusion_franka_fr3_softtoy/checkpoints/last/pretrained_model \
        --policy_type diffusion \
        --task "pick up the soft toy and place it in the drawer" \
        --fps 10

Example (with end-effector control):
    python deploy_robot_client.py \
        --use_sync_inference \
        --use_ee \
        --checkpoint_path outputs/train/diffusion_franka_fr3_ee_softtoy/checkpoints/last/pretrained_model \
        --policy_type diffusion \
        --task "pick and place task" \
        --fps 10
"""

import argparse
import json
import logging
from pathlib import Path

from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots.franka_fr3.config_franka_fr3 import FrankaFR3Config


def main():
    # Set up logger for main function
    logger = logging.getLogger("deploy_robot_client")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    logger.propagate = False
    
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
    
    # Robot control space
    parser.add_argument(
        "--use_ee",
        action="store_true",
        help="Use end-effector space instead of joint space (for policies trained on EE datasets)"
    )
    
    # Inference mode
    parser.add_argument(
        "--use_sync_inference", 
        action="store_true",
        help="Use synchronous inference (run policy locally) instead of async server. Recommended for diffusion policy until async issues are fixed."
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
    
    # Observation remapping
    parser.add_argument(
        "--rename_map",
        type=str,
        default="{}",
        help='JSON string to remap observation keys (e.g., \'{"observation.images.front_img": "observation.images.camera1"}\')'
    )
    
    args = parser.parse_args()
    
    # Parse rename_map from JSON string
    try:
        rename_map = json.loads(args.rename_map)
        if not isinstance(rename_map, dict):
            raise ValueError("rename_map must be a dictionary")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for rename_map: {e}")
    
    # Validate inputs
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    if not 0.0 <= args.chunk_size_threshold <= 1.0:
        raise ValueError(f"chunk_size_threshold must be between 0.0 and 1.0, got {args.chunk_size_threshold}")
    
    if args.actions_per_chunk <= 0:
        raise ValueError(f"actions_per_chunk must be positive, got {args.actions_per_chunk}")
    
    # Hardcoded camera configuration for Franka FR3
    base_camera_configs = {
        "front_img": RealSenseCameraConfig(
            serial_number_or_name="938422074102",
            width=640,
            height=480,
            fps=30,
            rotation=Cv2Rotation.ROTATE_180,
        ),
        "wrist_img": RealSenseCameraConfig(
            serial_number_or_name="919122070360",
            width=640,
            height=480,
            fps=30,
        ),
    }
    
    # Apply rename_map to camera keys if provided
    # This allows the robot to output observations with keys matching the policy's expectations
    camera_configs = {}
    for hw_camera_name, camera_config in base_camera_configs.items():
        # Build the full observation key as it would appear in observations
        full_key = f"observation.images.{hw_camera_name}"
        
        # Check if this key needs to be renamed
        if full_key in rename_map:
            # Extract the new camera name from the renamed key
            # e.g., "observation.images.camera1" -> "camera1"
            new_camera_name = rename_map[full_key].split(".")[-1]
            camera_configs[new_camera_name] = camera_config
            logger.info(f"Renamed camera '{hw_camera_name}' -> '{new_camera_name}' for policy compatibility")
        else:
            camera_configs[hw_camera_name] = camera_config
    
    # Create Franka FR3 robot configuration
    robot_config = FrankaFR3Config(
        id=args.robot_id,
        cameras=camera_configs,
        dt=1/args.fps,
        use_ee=args.use_ee
    )
    
    # Add safety parameters if provided
    if args.max_relative_target is not None:
        robot_config.max_relative_target = args.max_relative_target
    
    # Print configuration summary
    logger.info("="*70)
    logger.info("Franka FR3 Policy Deployment Client")
    logger.info("="*70)
    logger.info(f"Robot ID: {robot_config.id}")
    logger.info(f"Control Space: {'End-Effector (EE)' if args.use_ee else 'Joint Space'}")
    logger.info(f"Policy Type: {args.policy_type.upper()}")
    logger.info(f"Policy Checkpoint: {checkpoint_path}")
    logger.info(f"Policy Device: {args.policy_device}")
    logger.info(f"Task: {args.task or 'No task specified'}")
    logger.info(f"Cameras ({len(camera_configs)}):")
    for name, cfg in camera_configs.items():
        logger.info(f"  - {name}: {cfg.serial_number_or_name} @ {cfg.width}x{cfg.height} {cfg.fps}fps" + 
                   (f" (rotation: {cfg.rotation.value}°)" if hasattr(cfg, 'rotation') and cfg.rotation.value != 0 else ""))
    
    if not args.use_sync_inference:
        logger.info(f"Server Address: {args.server_address}")
        logger.info(f"Actions per Chunk: {args.actions_per_chunk}")
        logger.info(f"Chunk Size Threshold: {args.chunk_size_threshold}")
        logger.info(f"Aggregate Function: {args.aggregate_fn_name}")
    
    logger.info(f"FPS: {args.fps}")
    logger.info(f"Max Relative Target: {getattr(robot_config, 'max_relative_target', 'Not set')}")
    logger.info("="*70)
    
    if args.dry_run:
        logger.info("DRY RUN MODE: Configuration validated successfully!")
        logger.info("Remove --dry_run flag to actually connect to the robot.")
        return 0
    
    # Choose inference mode
    if args.use_sync_inference:
        logger.info("Using SYNCHRONOUS INFERENCE mode (policy runs locally)")
        return run_sync_inference(robot_config, checkpoint_path, args, logger)
    else:
        logger.info("Using ASYNC INFERENCE mode (policy runs on server)")
        return run_async_inference(robot_config, checkpoint_path, args, logger)


def run_async_inference(robot_config, checkpoint_path, args, logger):
    """Run with async inference (policy server)
       Note: Policies with n_obs_steps > 1 are not yet supported for Async inference
             as the policy server does not manage the observation queue properly. affected
             policies are (DP, VQ-BeT)
    
    Args:
        robot_config: Robot configuration
        checkpoint_path: Path to policy checkpoint
        args: Command line arguments
        logger: Logger instance to use
    """
    import threading
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.helpers import visualize_action_queue_size
    from lerobot.async_inference.robot_client import RobotClient
    
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


def add_resize_processor_if_needed(preprocessor, policy_config, robot_config, logger):
    """Add resize processor if camera dimensions don't match policy expectations.
    
    Args:
        preprocessor: The policy preprocessor pipeline
        policy_config: Policy configuration with input_features
        robot_config: Robot configuration with camera configs
        logger: Logger instance to use
    """
    from lerobot.processor.hil_processor import ImageCropResizeProcessorStep
    
    # Check if resize processor already exists
    has_resize_processor = any(
        isinstance(step, ImageCropResizeProcessorStep) for step in preprocessor.steps
    )
    
    if has_resize_processor:
        logger.info("Resize processor already present in preprocessor pipeline")
        return
    
    # Get expected image dimensions from policy config
    expected_dims = {}
    for key, feature in policy_config.input_features.items():
        if feature.type == "VISUAL" and "images" in key:
            camera_name = key.split(".")[-1]
            expected_dims[camera_name] = (feature.shape[1], feature.shape[2])  # (H, W)
    
    # Get actual camera dimensions from robot config
    needs_resize = False
    for camera_name, camera_config in robot_config.cameras.items():
        if camera_name in expected_dims:
            expected_h, expected_w = expected_dims[camera_name]
            actual_h, actual_w = camera_config.height, camera_config.width
            
            if (actual_h, actual_w) != (expected_h, expected_w):
                logger.info(
                    f"Camera '{camera_name}': actual size ({actual_w}x{actual_h}) != "
                    f"expected size ({expected_w}x{expected_h})"
                )
                needs_resize = True
    
    if not needs_resize:
        logger.info("Camera dimensions match policy expectations, no resize needed")
        return
    
    resize_size = next(iter(expected_dims.values()))  # All cameras should have same size
    resize_processor = ImageCropResizeProcessorStep(resize_size=resize_size)
    
    # Find insertion point (after to_batch, before device)
    insert_idx = None
    for idx, step in enumerate(preprocessor.steps):
        step_name = getattr(step.__class__, "_registry_name", "")
        if step_name == "to_batch_processor":
            insert_idx = idx + 1
            break
    
    if insert_idx is not None:
        preprocessor.steps.insert(insert_idx, resize_processor)
        logger.info(f"Added resize processor to pipeline at index {insert_idx} with size {resize_size}")
    else:
        # Fallback: insert at beginning if to_batch not found
        preprocessor.steps.insert(0, resize_processor)
        logger.info(f"Added resize processor at beginning of pipeline with size {resize_size}")


def run_sync_inference(robot_config, checkpoint_path, args, logger):
    """Run with synchronous inference (policy runs locally)
    
    Based on examples/tutorial/diffusion/diffusion_using_example.py
    
    Args:
        robot_config: Robot configuration
        checkpoint_path: Path to policy checkpoint
        args: Command line arguments
        logger: Logger instance to use
    """
    import time
    import torch
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.utils import build_inference_frame, make_robot_action
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.datasets.utils import hw_to_dataset_features
    from lerobot.utils.constants import ACTION, OBS_STR
    
    device = torch.device(args.policy_device)
    
    # Load policy
    logger.info(f"Loading {args.policy_type} policy from {checkpoint_path}...")
    policy = get_policy_class(args.policy_type).from_pretrained(str(checkpoint_path))
    
    if args.policy_type in ["pi05", "pi0"]:
        if hasattr(policy, 'model') and hasattr(policy.model, 'sample_actions'):
            # Disable torch.compile for PI05/PI0 to avoid long compilation time during inference
            # If sample_actions was compiled, replace it with the original uncompiled version
            # torch.compile wraps methods, we need to get the original
            if hasattr(policy.model.sample_actions, '__wrapped__'):
                policy.model.sample_actions = policy.model.sample_actions.__wrapped__
            logger.info("Disabled torch.compile for inference")
    
    policy.to(device)
    policy.eval()
    
    # Load preprocessor and postprocessor, overriding device to match requested device
    device_override = {"device": device}
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides={
            "device_processor": device_override,
        },
        postprocessor_overrides={"device_processor": device_override},
    )
    
    # Check if camera resolutions match policy expectations and add resize processor if needed
    add_resize_processor_if_needed(preprocess, policy.config, robot_config, logger)
    
    # Initialize robot
    logger.info("Connecting to robot...")
    robot = make_robot_from_config(robot_config)
    robot.connect()
    logger.info("Robot connected! Starting control loop (Ctrl+C to stop)...")
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    try:
        dt = 1.0 / args.fps
        while True:
            start_time = time.perf_counter()
            
            obs = robot.get_observation()
            obs_frame = build_inference_frame(
                observation=obs, ds_features=dataset_features, device=device
            )
            
            obs = preprocess(obs_frame)
            action = policy.select_action(obs)
            action = postprocess(action)
            action = make_robot_action(action, dataset_features)
            
            robot.send_action(action)
            
            # Maintain control frequency
            elapsed = time.perf_counter() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
        
    except KeyboardInterrupt:
        logger.info("Stopping robot...")
    finally:
        robot.disconnect()
        logger.info("Robot disconnected successfully.")
    
    return 0


if __name__ == "__main__":
    exit(main())

