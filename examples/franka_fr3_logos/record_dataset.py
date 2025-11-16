#!/usr/bin/env python3
"""
Record observations from Franka FR3 robot while teleoperation happens externally.

This script only records robot observations (joint states + camera images) without 
controlling the robot. You teleoperate the robot from a different package/interface,
and this script passively records what the robot is doing.

Usage:
    python record_dataset.py \
        --robot_id franka_fr3 \
        --dataset_repo_id franka_fr3/my_dataset \
        --dataset_root datasets/my_dataset \
        --task "Pick and place task" \
        --num_episodes 10 \
        --fps 30 \
        --image_size 96 96
"""

import argparse
import logging
import time
from threading import Lock, Thread

import cv2
import numpy as np
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Joy
from trajectory_msgs.msg import JointTrajectory

from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.robots.franka_fr3.config_franka_fr3 import FrankaFR3Config
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.constants import OBS_STR, ACTION


class RecordingController(Node):
    """ROS2 node to control recording via SpaceNav joystick."""
    
    def __init__(self):
        super().__init__('recording_controller')
        self.subscription = self.create_subscription(
            Joy,
            '/spacenav/joy',
            self.joy_callback,
            10
        )
        
        # Subscribe to joint trajectory commands
        self.joint_trajectory_subscription = self.create_subscription(
            JointTrajectory,
            '/fr3_arm_controller/joint_trajectory',
            self.joint_trajectory_callback,
            10
        )
        
        # Recording state: 'waiting', 'recording', 'save', or 'reject'
        self.state = 'waiting'
        self.lock = Lock()
        
        # Commanded joint positions (protected by mutex)
        self.commanded_positions = None
        self.command_lock = Lock()
        
        # Button indices
        self.BUTTON_A = 14  # Start recording
        self.BUTTON_B = 15  # Stop and save
        self.BUTTON_R = 8   # Reject episode
        
        self.get_logger().info("Recording controller initialized")
        self.get_logger().info("Press Button A to start recording")
        self.get_logger().info("Press Button B to stop and save episode")
        self.get_logger().info("Press Button R to reject episode (discard without saving)")
    
    def joy_callback(self, msg):
        """Handle joystick messages from SpaceNav."""
        # Ignore messages without correct button count
        if len(msg.buttons) != 27:
            return
        with self.lock:
            # Button A: Start recording (only if waiting)
            if msg.buttons[self.BUTTON_A] == 1 and self.state == 'waiting':
                self.state = 'recording'
                self.get_logger().info("▶ STARTING RECORDING")
            
            # Button B: Stop recording and save (only if recording)
            elif msg.buttons[self.BUTTON_B] == 1 and self.state == 'recording':
                self.state = 'save'
                self.get_logger().info("■ STOPPING RECORDING - Will save episode")
            
            # Button R: Reject episode (only if recording)
            elif msg.buttons[self.BUTTON_R] == 1 and self.state == 'recording':
                self.state = 'reject'
                self.get_logger().info("✗ REJECTING EPISODE - Will discard without saving")
    
    def joint_trajectory_callback(self, msg):
        """Handle joint trajectory command messages."""
        if msg.points and len(msg.points) > 0:
            # Store the commanded positions from the first trajectory point
            with self.command_lock:
                self.commanded_positions = list(msg.points[0].positions)
    
    def is_waiting_to_start(self):
        """Check if waiting for Button A to start recording."""
        with self.lock:
            return self.state == 'waiting'
    
    def is_recording(self):
        """Check if currently recording."""
        with self.lock:
            return self.state == 'recording'
    
    def should_save(self):
        """Check if recording should stop and save."""
        with self.lock:
            return self.state == 'save'
    
    def should_reject(self):
        """Check if recording should stop and reject (discard)."""
        with self.lock:
            return self.state == 'reject'
    
    def reset_to_waiting(self):
        """Reset state to waiting after episode is saved."""
        with self.lock:
            self.state = 'waiting'
    
    def get_commanded_positions(self):
        """Get the latest commanded positions (thread-safe)."""
        with self.command_lock:
            return self.commanded_positions.copy() if self.commanded_positions is not None else None


def record_episode(robot, dataset, fps, task, controller, image_size=None):
    """Record a single episode of observations.
    
    The action at time t is the observation at time t+1 (next state).
    This is the standard format for imitation learning datasets.
    Recording is controlled by SpaceNav buttons A (start) and B (stop/save).
    
    Args:
        robot: Robot instance
        dataset: LeRobotDataset instance
        fps: Recording frequency
        task: Task description
        controller: RecordingController instance
        image_size: Optional tuple (height, width) to resize images
    """
    print(f"\n{'='*70}")
    print(f"Waiting for Button A('3') to start recording episode {dataset.num_episodes + 1}")
    print(f"Recording at {fps} fps")
    print(f"{'='*70}\n")
    
    # Wait for Button A to start recording
    while controller.is_waiting_to_start():
        time.sleep(0.01)
    
    print(f"\n{'='*70}")
    print(f"🔴 RECORDING EPISODE {dataset.num_episodes}")
    print(f"Press Button B('4') to stop and save")
    print(f"Press Button R(square button on right) to reject and discard")
    print(f"{'='*70}\n")
    
    dt = 1.0 / fps
    start_time = time.perf_counter()
    frame_count = 0
    
    # Keep track of previous observation frame
    prev_obs_frame = None
    
    try:
        while controller.is_recording():
            loop_start = time.perf_counter()
            
            # Get current observation from robot
            obs = robot.get_observation()
            
            # Get commanded positions from controller
            commanded_positions = controller.get_commanded_positions()
            
            # Strip .pos suffix from observation keys
            obs = {key.removesuffix(".pos"): value for key, value in obs.items()}
            
            # Resize images if image_size is specified
            if image_size is not None:
                for key in obs:
                    if key.endswith("_img") or key.endswith("_image"):
                        obs[key] = cv2.resize(obs[key], (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
            
            # Build observation frame
            obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            
            # Add commanded positions to observation frame if available
            if commanded_positions is not None:
                obs_frame["observation.command"] = np.array(commanded_positions, dtype=np.float32)
            
            # For action, use the next observation (current obs becomes action for previous frame)
            if prev_obs_frame is not None:
                # The action at time t is the observation at time t+1
                action_frame = build_dataset_frame(dataset.features, obs, prefix=ACTION)
                
                # Combine previous observation with current action (next state)
                frame = {**prev_obs_frame, **action_frame, "task": task}
                dataset.add_frame(frame)
                frame_count += 1
            
            # Store current observation frame for next iteration
            prev_obs_frame = obs_frame
            
            # Maintain control frequency
            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
            else:
                logging.warning(f"Loop took {elapsed:.3f}s, longer than target {dt:.3f}s")
        
        # Check if episode was rejected
        if controller.should_reject():
            print(f"\n✗ Episode rejected! Recorded {frame_count} frames (discarded)")
            controller.reset_to_waiting()
            return False  # Return False to indicate rejection
        
        # Handle the last frame: duplicate the last observation as action
        if prev_obs_frame is not None:
            # Use the last observation as both obs and action for the final frame
            action_frame = build_dataset_frame(dataset.features, obs, prefix=ACTION)
            frame = {**prev_obs_frame, **action_frame, "task": task}
            dataset.add_frame(frame)
            frame_count += 1
        
        print(f"\n✓ Recorded {frame_count} frames in {time.perf_counter() - start_time:.2f}s")
        controller.reset_to_waiting()
        return True
        
    except KeyboardInterrupt:
        print(f"\n\n⚠ Episode interrupted by Ctrl+C! Recorded {frame_count} frames.")
        return False


def main():
    # Set up logger for main function
    logger = logging.getLogger("record_dataset")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    logger.propagate = False
    
    parser = argparse.ArgumentParser(
        description="Record observations from Franka FR3 and create LeRobot dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Robot configuration
    parser.add_argument("--robot_id", type=str, default="franka_fr3", 
                       help="Robot identifier")
    
    # Dataset configuration  
    parser.add_argument("--dataset_repo_id", type=str, required=True,
                       help="Dataset repository ID (e.g., username/dataset_name)")
    parser.add_argument("--dataset_root", type=str, required=True,
                       help="Local directory to save dataset")
    parser.add_argument("--task", type=str, required=True,
                       help="Task description")
    
    # Recording parameters
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=30,
                       help="Recording frequency (frames per second)")
    parser.add_argument("--image_size", type=int, nargs=2, default=[96, 96],
                       help="Image size (height width) to resize captured images (default: 96 96)")
    
    # Optional parameters
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Upload dataset to Hugging Face Hub after recording")
    parser.add_argument("--resume", action="store_true",
                       help="Resume recording on existing dataset")
    
    args = parser.parse_args()
    
    # Camera configuration for Franka FR3
    camera_configs = {
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
    
    # Create robot configuration
    robot_config = FrankaFR3Config(
        id=args.robot_id,
        cameras=camera_configs,
    )
    
    logger.info("="*70)
    logger.info("Franka FR3 Observation-Only Recording with SpaceNav Control")
    logger.info("="*70)
    logger.info(f"Dataset: {args.dataset_repo_id}")
    logger.info(f"Local path: {args.dataset_root}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Episodes: {args.num_episodes}")
    logger.info(f"FPS: {args.fps}")
    logger.info(f"Image size: {args.image_size[0]}x{args.image_size[1]}")
    logger.info(f"Cameras: {list(camera_configs.keys())}")
    logger.info("="*70)
    
    # Initialize robot (read-only mode) - robot will init ROS2
    logger.info("Connecting to robot...")
    robot = make_robot_from_config(robot_config)
    robot.connect()
    logger.info("Robot connected!")
    
    # Initialize recording controller (ROS2 already initialized by robot)
    controller = RecordingController()
    
    # Create executor and spin in separate thread
    executor = SingleThreadedExecutor()
    executor.add_node(controller)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    logger.info("ROS2 node spinning in separate thread")
    
    # Get features from robot configuration
    action_features = hw_to_dataset_features(robot.action_features, ACTION, use_video=True)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=True)
    
    # Strip .pos suffix from feature names
    for feature_key in action_features:
        if "names" in action_features[feature_key] and isinstance(action_features[feature_key]["names"], list):
            action_features[feature_key]["names"] = [
                name.removesuffix(".pos") for name in action_features[feature_key]["names"]
            ]
    
    for feature_key in obs_features:
        if "names" in obs_features[feature_key] and isinstance(obs_features[feature_key]["names"], list):
            obs_features[feature_key]["names"] = [
                name.removesuffix(".pos") for name in obs_features[feature_key]["names"]
            ]
    
    # Override image shapes with custom size
    for feature_key in obs_features:
        if obs_features[feature_key].get("dtype") == "video":
            # Update shape to custom image size (channels, height, width)
            obs_features[feature_key]["shape"] = (3, args.image_size[0], args.image_size[1])
            logger.info(f"Overriding {feature_key} shape to {obs_features[feature_key]['shape']}")

    # Add command feature to observations (commanded joint positions from controller)
    # The command feature only has 7 arm joints (no gripper)
    command_feature = {
        "observation.command": {
            "dtype": "float32",
            "shape": (len(robot.joint_names) - 1,),
            "names": robot.joint_names[:-1],
        }
    }
    
    features = {**action_features, **obs_features, **command_feature}
    
    try:
        # Create or load dataset
        if args.resume:
            logger.info(f"Resuming dataset from {args.dataset_root}")
            dataset = LeRobotDataset(
                args.dataset_repo_id,
                root=args.dataset_root,
            )
            if hasattr(robot, 'cameras') and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=0,
                    num_threads=4 * len(robot.cameras),
                )
        else:
            logger.info("Creating new dataset...")
            dataset = LeRobotDataset.create(
                args.dataset_repo_id,
                args.fps,
                root=args.dataset_root,
                robot_type=robot.name,
                features=features,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=4 * len(robot.cameras),
            )
        
        with VideoEncodingManager(dataset):
            episodes_recorded = 0
            
            while episodes_recorded < args.num_episodes:
                try:
                    # Record episode (controlled by SpaceNav buttons)
                    success = record_episode(
                        robot=robot,
                        dataset=dataset,
                        fps=args.fps,
                        task=args.task,
                        controller=controller,
                        image_size=args.image_size
                    )
                    
                    if success:
                        dataset.save_episode()
                        episodes_recorded += 1
                        logger.info(f"✓ Episode saved! ({episodes_recorded}/{args.num_episodes})")
                        
                        if episodes_recorded < args.num_episodes:
                            print(f"\nReset the environment for next episode.")
                            print(f"Press Button A when ready to record episode {episodes_recorded + 1}")
                            print("Or press Ctrl+C to finish recording.\n")
                    else:
                        logger.info("Episode rejected - clearing buffer")
                        dataset.clear_episode_buffer()
                        print(f"\nPress Button A to start a fresh recording of episode {episodes_recorded + 1}")
                        print("Or press Ctrl+C to finish recording.\n")
                        
                except KeyboardInterrupt:
                    print("\n\n⚠ Stopping recording...")
                    break
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Recording complete! {episodes_recorded} episodes saved.")
        logger.info(f"{'='*70}\n")
        
        dataset.finalize()
        logger.info("Dataset finalized!")
        
        # Upload to hub if requested
        if args.push_to_hub:
            logger.info("Uploading dataset to Hugging Face Hub...")
            dataset.push_to_hub()
            logger.info("Upload complete!")
        
    finally:
        logger.info("Shutting down recording controller...")
        try:
            executor.shutdown()
            controller.destroy_node()
            robot.disconnect()
            logger.info("Robot disconnected")
        except Exception as e:
            logger.warning(f"Error during executor cleanup: {e}")
    
    return 0


if __name__ == "__main__":
    exit(main())
