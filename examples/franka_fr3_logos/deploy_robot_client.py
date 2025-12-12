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
        default="",
        choices=["act", "diffusion", "vqbet", "smolvla", "groot", "pi0", "pi05", "xvla", ""],
        help="Type of policy to use (can be provided via command in interactive mode)"
    )
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default="",
        help="Path to the trained policy checkpoint (can be provided via command in interactive mode)"
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
    
    # Rollout parameters
    parser.add_argument(
        "--max_rollout_steps",
        type=int,
        default=None,
        help="Maximum number of rollout steps (None for unlimited)"
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
    
    # Interactive mode (ROS2 subscriber)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run as ROS2 subscriber for interactive rollout control"
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
    
    # Validate inputs (skip checkpoint validation in interactive mode)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    if not args.interactive:
        if not checkpoint_path or not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {args.checkpoint_path}")
    
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
    logger.info(f"Policy Checkpoint: {checkpoint_path or 'Will be provided via command'}")
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
    logger.info(f"Max Rollout Steps: {args.max_rollout_steps if args.max_rollout_steps else 'Unlimited'}")
    logger.info(f"Max Relative Target: {getattr(robot_config, 'max_relative_target', 'Not set')}")
    logger.info("="*70)
    
    if args.dry_run:
        logger.info("DRY RUN MODE: Configuration validated successfully!")
        logger.info("Remove --dry_run flag to actually connect to the robot.")
        return 0
    
    # Interactive mode: run as ROS2 subscriber
    if args.interactive:
        logger.info("Running in INTERACTIVE mode (ROS2 subscriber)")
        return run_interactive_server(robot_config, args, logger)
    
    # Choose inference mode
    if args.use_sync_inference:
        logger.info("Using SYNCHRONOUS INFERENCE mode (policy runs locally)")
        ret_code, sync_state = run_sync_inference(robot_config, checkpoint_path, args, logger)
        # Disconnect robot in non-interactive mode
        if sync_state and sync_state.get('robot'):
            sync_state['robot'].disconnect()
            logger.info("Robot disconnected successfully.")
        return ret_code
    else:
        logger.info("Using ASYNC INFERENCE mode (policy runs on server)")
        return run_async_inference(robot_config, checkpoint_path, args, logger)[0]


def run_async_inference(robot_config, checkpoint_path, args, logger, stop_event=None, client=None):
    """Run with async inference (policy server)
       Note: Policies with n_obs_steps > 1 are not yet supported for Async inference
             as the policy server does not manage the observation queue properly. affected
             policies are (DP, VQ-BeT)
    
    Args:
        robot_config: Robot configuration
        checkpoint_path: Path to policy checkpoint
        args: Command line arguments
        logger: Logger instance to use
        stop_event: Optional threading.Event to signal early termination
        client: Optional pre-created RobotClient (for caching in interactive mode)
    
    Returns:
        Tuple of (return_code, client) for caching
    """
    import threading
    import time
    from lerobot.async_inference.helpers import visualize_action_queue_size
    
    # Create client if not provided (non-interactive mode)
    owns_client = client is None
    if client is None:
        from lerobot.async_inference.configs import RobotClientConfig
        from lerobot.async_inference.robot_client import RobotClient
        
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
            max_rollout_steps=args.max_rollout_steps,
        )
        client = RobotClient(client_config)
    
    try:
        # Start client if we created it (owns_client) or if it's not running
        if owns_client:
            logger.info("Connecting to policy server and robot...")
            if not client.start():
                logger.error("Failed to start robot client")
                return 1, client
            logger.info("Successfully connected!")
        else:
            logger.info("Reusing existing connection to policy server and robot...")
        
        logger.info("Starting control loop... Press Ctrl+C to stop execution.")
        
        # Start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()
        
        # Monitor thread to handle stop signal
        def stop_monitor():
            while client.running:
                if stop_event and stop_event.is_set():
                    logger.info("Stop signal received, stopping client...")
                    client.stop()
                time.sleep(0.1)
        
        if stop_event:
            threading.Thread(target=stop_monitor, daemon=True).start()
        
        # Run the control loop (blocking)
        client.control_loop(args.task)
        
        # Normal completion (e.g., max_rollout_steps reached)
        logger.info("Control loop completed.")
        # Only stop client if we own it AND not in interactive mode (stop_event indicates interactive mode)
        if owns_client and client.running and stop_event is None:
            client.stop()
        else:
            # For reused clients, stop the threads but keep connection alive
            # This ensures the action_receiver_thread exits cleanly before we start a new one
            client.shutdown_event.set()
        action_receiver_thread.join(timeout=5.0)
        
        # Optionally visualize queue size statistics
        if args.debug_visualize_queue_size and hasattr(client, 'action_queue_size'):
            logger.info("Displaying action queue size visualization...")
            visualize_action_queue_size(client.action_queue_size)
        
        if owns_client and stop_event is None:
            logger.info("Robot client stopped successfully.")
        
    except KeyboardInterrupt:
        logger.info("Stopping robot client...")
        if client.running and stop_event is None:
            client.stop()
        action_receiver_thread.join(timeout=5.0)
        
        # Optionally visualize queue size statistics
        if args.debug_visualize_queue_size and hasattr(client, 'action_queue_size'):
            logger.info("Displaying action queue size visualization...")
            visualize_action_queue_size(client.action_queue_size)
        
        if stop_event is None:
            logger.info("Robot client stopped successfully.")
        
    except Exception as e:
        logger.error(f"Robot client error: {e}")
        if client.running and stop_event is None:
            client.stop()
        return 1, client
    
    return 0, client


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


def run_sync_inference(robot_config, checkpoint_path, args, logger, stop_event=None, sync_state=None):
    """Run with synchronous inference (policy runs locally)
    
    Based on examples/tutorial/diffusion/diffusion_using_example.py
    
    Args:
        robot_config: Robot configuration
        checkpoint_path: Path to policy checkpoint
        args: Command line arguments
        logger: Logger instance to use
        stop_event: Optional threading.Event to signal early termination
        sync_state: Optional dict with cached state {'policy', 'preprocess', 'postprocess', 'robot', 'dataset_features'}
    
    Returns:
        Tuple of (return_code, sync_state) for caching
    """
    import time
    import torch
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.utils import build_inference_frame, make_robot_action
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.datasets.utils import hw_to_dataset_features
    from lerobot.utils.constants import ACTION, OBS_STR
    
    device = torch.device(args.policy_device)
    
    # Use cached state or create new
    if sync_state is None:
        sync_state = {}
    
    policy = sync_state.get('policy')
    preprocess = sync_state.get('preprocess')
    postprocess = sync_state.get('postprocess')
    robot = sync_state.get('robot')
    dataset_features = sync_state.get('dataset_features')
    
    # Load policy if not provided
    if policy is None:
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
    
    # Initialize robot if not provided
    if robot is None:
        logger.info("Connecting to robot...")
        robot = make_robot_from_config(robot_config)
        robot.connect()
        logger.info("Robot connected!")
        action_features = hw_to_dataset_features(robot.action_features, ACTION)
        obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
        dataset_features = {**action_features, **obs_features}
    
    logger.info("Starting control loop (Ctrl+C to stop)...")

    try:
        dt = 1.0 / args.fps
        step = 0
        while args.max_rollout_steps is None or step < args.max_rollout_steps:
            if stop_event and stop_event.is_set():
                logger.info("Stop signal received, ending rollout...")
                break
            
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
            
            step += 1
        
        if args.max_rollout_steps is not None:
            logger.info(f"Completed {step} rollout steps.")
        
    except KeyboardInterrupt:
        logger.info("Stopping robot...")
    
    # Build sync_state for caching (don't disconnect robot if caching)
    sync_state = {
        'policy': policy,
        'preprocess': preprocess,
        'postprocess': postprocess,
        'robot': robot,
        'dataset_features': dataset_features,
    }
    
    return 0, sync_state


def run_interactive_server(robot_config, args, logger):
    """Run as ROS2 subscriber for interactive rollout control."""
    import json
    import threading
    from argparse import Namespace
    from pathlib import Path
    
    import rclpy
    import time
    import yaml
    from rclpy.node import Node
    from std_msgs.msg import String, Empty
    from sensor_msgs.msg import Joy

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from replay_h5 import replay_h5_on_robot
    
    class RolloutSubscriber(Node):
        def __init__(self):
            super().__init__('rollout_subscriber')
            self.create_subscription(String, '/rollout_command', self.on_command, 10)
            self.create_subscription(Joy, '/spacenav/joy', self.joy_callback, 10)
            self.reset_pub = self.create_publisher(Empty, '/franka_reset', 10)
            logger.info("Listening for rollout commands on '/rollout_command'...")
            
            # Load model configurations from external config file
            config_file = Path(__file__).parent / "rollout_model_configs.yaml"
            with open(config_file) as f:
                self.model_id_to_config = yaml.safe_load(f)
            logger.info(f"Loaded {len(self.model_id_to_config)} rollout model configs from {config_file}")
            
            self.running = False
            self.awaiting_result = False
            self.result_file = None
            self.button_pressed = None
            self.stop_event = threading.Event()
            
            # Sync mode cache
            self.cached_sync_checkpoint_path = None
            self.cached_sync_state = None
            # Async mode cache
            self.cached_async_checkpoint_path = None
            self.cached_async_client = None
        
        def joy_callback(self, msg):
            if len(msg.buttons) < 27:
                return
            
            # spacemouse right square button stops inference early (if running)
            if msg.buttons[8] and self.running and not self.awaiting_result:
                self.stop_event.set()
            
            # spacemouse button 1 / 2 for success/failure feedback
            if self.awaiting_result:
                if msg.buttons[12]:
                    self.button_pressed = True
                elif msg.buttons[13]:
                    self.button_pressed = False
    
        def on_command(self, msg):
            if self.running:
                logger.warning("Rollout already in progress, ignoring command")
                return
            
            self.running = True
            data = json.loads(msg.data)
            
            model_id = data.get("model_id")
            model_config = self.model_id_to_config.get(model_id)
            policy_type = model_config.get("policy_type")
            mode = model_config.get("mode") if model_config else None
            checkpoint_path = model_config.get("checkpoint_path")
            task = data.get("description") or args.task
            max_steps = args.max_rollout_steps or 0
            
            if not checkpoint_path or not task:
                logger.error("Missing checkpoint_path or task")
                self.running = False
                return
            
            logger.info(f"Rollout: model_id={model_id}, checkpoint={checkpoint_path}, task='{task}', max_steps={max_steps or 'unlimited'}")
            
            rollout_args = Namespace(**vars(args))
            rollout_args.policy_type = policy_type
            rollout_args.checkpoint_path = checkpoint_path
            rollout_args.task = task
            rollout_args.max_rollout_steps = max_steps if max_steps > 0 else None
            
            self.result_file = Path(f'logs/results/{data.get("metadata")}_{model_id}_result.log')
            self.result_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Run inference in a separate thread so joy callbacks keep working
            self.stop_event.clear()
            inference_fn = run_sync_inference if mode == "sync" else run_async_inference
            inference_thread = threading.Thread(
                target=self._run_inference,
                args=(inference_fn, robot_config, checkpoint_path, rollout_args, mode == "replay"),
                daemon=True
            )
            inference_thread.start()
        
        def _run_inference(self, inference_fn, robot_config, checkpoint_path, rollout_args, is_replay=False):
            # NOTE: Currently, mixing multiple policies in a single interactive session is not supported.
            # The cache is separate for each mode, but switching between policies may cause issues with robot/server state.
            # This feature will be implemented in a future update.
            try:
                if is_replay:
                    replay_h5_on_robot(h5_path=checkpoint_path, robot_id=args.robot_id, logger=logger)
                elif inference_fn == run_sync_inference:
                    # For sync mode, check cache and pass to inference function
                    sync_state = None
                    if self.cached_sync_checkpoint_path == checkpoint_path and self.cached_sync_state is not None:
                        logger.info(f"Reusing cached sync state from {checkpoint_path}")
                        self.cached_sync_state['policy'].reset()  # Reset internal state for new rollout
                        sync_state = self.cached_sync_state
                    
                    _, sync_state = inference_fn(
                        robot_config, Path(checkpoint_path), rollout_args, logger, self.stop_event,
                        sync_state=sync_state)
                    
                    # Update cache
                    self.cached_sync_checkpoint_path = checkpoint_path
                    self.cached_sync_state = sync_state
                else:
                    # For async mode, check cache and pass to inference function
                    client = None
                    if self.cached_async_checkpoint_path == checkpoint_path and self.cached_async_client is not None:
                        logger.info(f"Reusing cached async client for {checkpoint_path}")
                        client = self.cached_async_client
                        client.reset()
                    
                    _, client = run_async_inference(
                        robot_config, Path(checkpoint_path), rollout_args, logger, self.stop_event,
                        client=client)
                    
                    # Update cache
                    self.cached_async_checkpoint_path = checkpoint_path
                    self.cached_async_client = client
            except Exception as e:
                logger.error(f"{'Replay' if is_replay else 'Inference'} error: {e}")
            
            # Wait for user feedback via spacenav button
            logger.info("Press spacemouse button 1 for success, button 2 for failure...")
            self.awaiting_result = True
            self.button_pressed = None
            while self.button_pressed is None:
                time.sleep(0.1)
            
            result = "success" if self.button_pressed else "failure"
            with open(self.result_file, "a") as f:
                f.write(checkpoint_path + "\n")
                f.write(result + "\n")
            logger.info(f"Logged result: {result}")
            self.reset_pub.publish(Empty())
            logger.info("Robot reset command sent... Will wait for 3 secs")
            time.sleep(3)
            self.awaiting_result = False
            self.running = False
    
    rclpy.init()
    try:
        rclpy.spin(RolloutSubscriber())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    exit(main())

