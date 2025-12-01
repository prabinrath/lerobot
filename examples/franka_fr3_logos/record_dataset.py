#!/usr/bin/env python3

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
Record observations from Franka FR3 robot to H5 format while teleoperation happens externally.

This script only records robot observations (joint states, command, camera images) without 
controlling the robot. You teleoperate the robot from a different package/interface,
and this script passively records what the robot is doing.

The H5 format follows this structure:
- data/
  - demo_0/
    - front_img: (T, H, W, 3) uint8
    - wrist_img: (T, H, W, 3) uint8
    - joint_states: (T, 3, 8) float32 - [position, velocity, acceleration] x 8 joints
    - command: (T, 7) float32 - commanded joint positions (7 arm joints, no gripper)

Usage:
    python record_dataset.py \
        --robot_id franka_fr3 \
        --output_path datasets/my_dataset.h5 \
        --num_episodes 10 \
        --fps 30 \
        --image_size 96 96
"""

import argparse
import logging
import time
from pathlib import Path
from threading import Lock, Thread

import cv2
import h5py
import numpy as np
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory

from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots.franka_fr3.config_franka_fr3 import FrankaFR3Config
from lerobot.robots.utils import make_robot_from_config


class RecordingController(Node):
    """ROS2 node to control recording via SpaceNav joystick or ROS2 commands."""
    
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
        
        # Subscribe to record commands from interactive server
        self.record_command_subscription = self.create_subscription(
            String,
            '/record_command',
            self.record_command_callback,
            10
        )
        
        # Recording state: 'waiting', 'recording', 'save', or 'reject'
        self.state = 'waiting'
        self.lock = Lock()
        
        # Metadata from start command
        self.metadata = None
        
        # Commanded joint positions (protected by mutex)
        self.commanded_positions = None
        self.command_lock = Lock()
        
        # Button indices
        self.BUTTON_A = 14  # Start recording
        self.BUTTON_B = 15  # Stop and save
        self.BUTTON_R = 8   # Reject episode
        
        self.get_logger().info("Recording controller initialized")
        self.get_logger().info("Press Button A('3') to start recording")
        self.get_logger().info("Press Button B('4') to stop and save episode")
        self.get_logger().info("Press Button R to reject episode (discard without saving)")
        self.get_logger().info("Or use /record_command topic: 'start' / 'stop'")
    
    def record_command_callback(self, msg):
        """Handle record commands from interactive server."""
        with self.lock:
            if msg.data.startswith("start:") and self.state == 'waiting':
                self.metadata = msg.data[6:]  # Extract metadata after "start:"
                self.state = 'recording'
                self.get_logger().info(f"▶ STARTING RECORDING (via command) - metadata: {self.metadata}")
            elif msg.data == "stop" and self.state == 'recording':
                self.state = 'save'
                self.get_logger().info("■ STOPPING RECORDING (via command) - Will save episode")
    
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
    
    def get_metadata(self):
        """Get the metadata from the last start command."""
        with self.lock:
            return self.metadata


def record_episode(robot, episode_idx, fps, controller, image_size=None):
    """Record a single episode of observations to H5 format.
    
    Recording is controlled by SpaceNav buttons A (start) and B (stop/save).
    
    Args:
        robot: Robot instance
        episode_idx: Current episode index
        fps: Recording frequency
        controller: RecordingController instance
        image_size: Optional tuple (height, width) to resize images
        
    Returns:
        Dictionary with episode data or None if rejected:
        {
            'front_img': (T, H, W, 3) numpy array,
            'wrist_img': (T, H, W, 3) numpy array,
            'joint_states': (T, 3, 8) numpy array - [position, velocity, acceleration] x 8 joints,
            'command': (T, 7) numpy array - commanded joint positions (arm joints only),
            'timestamp': (T,) numpy array - timestamps for each frame
        }
    """
    print(f"\n{'='*70}")
    print(f"Waiting for Button A('3') to start recording episode {episode_idx + 1}")
    print(f"Recording at {fps} fps")
    print(f"{'='*70}\n")
    
    # Wait for Button A to start recording
    while controller.is_waiting_to_start():
        time.sleep(0.01)
    
    print(f"\n{'='*70}")
    print(f"🔴 RECORDING EPISODE {episode_idx + 1}")
    print(f"Press Button B('4') to stop and save")
    print(f"Press Button R(square button on right) to reject and discard")
    print(f"{'='*70}\n")
    
    dt = 1.0 / fps
    start_time = time.perf_counter()
    
    # Buffers to store episode data
    front_imgs = []
    wrist_imgs = []
    joint_states = []  # Will store (3, 8) arrays: [position, velocity, acceleration] x 8 joints
    commands = []  # Will store commanded joint positions (7 arm joints)
    timestamps = []  # Will store timestamps for each frame
    
    try:
        while controller.is_recording():
            loop_start = time.perf_counter()
            
            # Record timestamp (relative to start of episode)
            current_timestamp = time.perf_counter() - start_time
            
            # Get current observation from robot
            obs = robot.get_observation()
            
            # Get commanded positions from controller
            commanded_positions = controller.get_commanded_positions()
            
            # Strip .pos suffix from observation keys
            obs = {key.removesuffix(".pos"): value for key, value in obs.items()}
            
            # Extract and resize camera images
            front_img = obs.get('front_img')
            wrist_img = obs.get('wrist_img')
            
            if image_size is not None:
                if front_img is not None:
                    front_img = cv2.resize(front_img, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
                if wrist_img is not None:
                    wrist_img = cv2.resize(wrist_img, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
            
            # Extract joint state information
            # Robot returns joint positions in obs with joint names
            joint_names = robot.joint_names  # Should be 8 joints: 7 arm joints + gripper
            
            # Create joint_states array: shape (3, 8) for [position, velocity, acceleration] x 8 joints
            joint_state = np.zeros((3, 8), dtype=np.float32)
            
            # Fill in positions (first row)
            for i, joint_name in enumerate(joint_names):
                if joint_name in obs:
                    joint_state[0, i] = obs[joint_name]
            
            # Velocity and acceleration would go in rows 1 and 2 if available
            # For now, they remain zero as robot.get_observation() typically only provides positions
            # Store commanded positions (7 arm joints, no gripper)
            if commanded_positions is not None:
                # commanded_positions should have 7 values for the arm joints
                command = np.array(commanded_positions[:7], dtype=np.float32)
            else:
                # If no command available, use zeros
                command = np.zeros(7, dtype=np.float32)
            
            # Store data
            front_imgs.append(front_img)
            wrist_imgs.append(wrist_img)
            joint_states.append(joint_state)
            commands.append(command)
            timestamps.append(current_timestamp)
            
            # Maintain control frequency
            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
            else:
                logging.warning(f"Loop took {elapsed:.3f}s, longer than target {dt:.3f}s")
        
        # Check if episode was rejected
        if controller.should_reject():
            print(f"\n✗ Episode rejected! Recorded {len(front_imgs)} frames (discarded)")
            controller.reset_to_waiting()
            return None  # Return None to indicate rejection
        
        print(f"\n✓ Recorded {len(front_imgs)} frames in {time.perf_counter() - start_time:.2f}s")
        controller.reset_to_waiting()
        
        # Convert lists to numpy arrays
        return {
            'front_img': np.array(front_imgs, dtype=np.uint8),
            'wrist_img': np.array(wrist_imgs, dtype=np.uint8),
            'joint_states': np.array(joint_states, dtype=np.float32),
            'command': np.array(commands, dtype=np.float32),
            'timestamp': np.array(timestamps, dtype=np.float64)
        }
        
    except KeyboardInterrupt:
        print(f"\n\n⚠ Episode interrupted by Ctrl+C! Recorded {len(front_imgs)} frames.")
        return None


def save_episode_to_h5(h5_file, episode_data, episode_idx):
    """
    Save episode data to H5 file.
    
    Args:
        h5_file: Open h5py.File object
        episode_data: Dictionary with 'front_img', 'wrist_img', 'joint_states', 'command', 'timestamp'
        episode_idx: Episode index
    """
    demo_name = f"demo_{episode_idx}"
    demo_group = h5_file['data'].create_group(demo_name)
    
    # Save data
    demo_group.create_dataset('front_img', data=episode_data['front_img'], compression='gzip')
    demo_group.create_dataset('wrist_img', data=episode_data['wrist_img'], compression='gzip')
    demo_group.create_dataset('joint_states', data=episode_data['joint_states'], compression='gzip')
    demo_group.create_dataset('command', data=episode_data['command'], compression='gzip')
    demo_group.create_dataset('timestamp', data=episode_data['timestamp'], compression='gzip')
    
    # Flush to ensure data is written
    h5_file.flush()


def main():
    # Set up logger for main function
    logger = logging.getLogger("record_dataset")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    logger.propagate = False
    
    parser = argparse.ArgumentParser(
        description="Record observations from Franka FR3 and save to H5 format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Robot configuration
    parser.add_argument("--robot_id", type=str, default="franka_fr3", 
                       help="Robot identifier")
    
    # Output configuration  
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save H5 dataset file (e.g., datasets/my_dataset.h5)")
    
    # Recording parameters
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=30,
                       help="Recording frequency (frames per second)")
    parser.add_argument("--image_size", type=int, nargs=2, default=[96, 96],
                       help="Image size (height width) to resize captured images (default: 96 96)")
    
    # Optional parameters
    parser.add_argument("--resume", action="store_true",
                       help="Resume recording on existing H5 file")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    logger.info("Franka FR3 Data Recording")
    logger.info("="*70)
    logger.info(f"Output: {args.output_path} [can be overridden by record metadata]")
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
    
    try:
        start_episode = 0
        current_episode = start_episode
        resumed_files = set()  # Track files we've already resumed from
        
        # Check if output file exists and resume flag is not set (only for spacemouse recording)
        # NOTE: When recording via ROS2 command with metadata, resume is automatic and doesn't need the flag
        output_path = Path(args.output_path)
        if output_path.exists() and not args.resume:
            logger.error(f"Dataset file already exists: {output_path}")
            logger.error("Use --resume flag to continue recording, or delete/rename the existing file.")
            return 1
        
        while (current_episode - start_episode) < args.num_episodes:
            try:
                # Record episode (controlled by SpaceNav buttons)
                episode_data = record_episode(
                    robot=robot,
                    episode_idx=current_episode,
                    fps=args.fps,
                    controller=controller,
                    image_size=args.image_size
                )
                
                if episode_data is not None:
                    # Get metadata from controller and use as h5 filename
                    metadata = controller.get_metadata()
                    if metadata:
                        output_path = Path(args.output_path).parent / f"{metadata}.h5"
                    else:
                        output_path = Path(args.output_path)
                    
                    # Create file if it doesn't exist, otherwise resume from existing
                    if not output_path.exists():
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with h5py.File(output_path, 'w') as f:
                            f.create_group('data')
                            f.attrs['fps'] = args.fps
                            f.attrs['image_height'] = args.image_size[0]
                            f.attrs['image_width'] = args.image_size[1]
                        logger.info(f"Created new H5 file: {output_path}")
                        resumed_files.add(str(output_path))
                    elif str(output_path) not in resumed_files:
                        # Resume from existing file - validate and get next episode index (only once per file)
                        logger.info(f"Resuming from existing file: {output_path}")
                        logger.info("If you don't want to resume, provide a new file name / metadata or delete the existing H5 file.")
                        with h5py.File(output_path, 'r') as f:
                            # Verify metadata matches
                            if 'fps' in f.attrs and f.attrs['fps'] != args.fps:
                                logger.error(f"Existing FPS ({f.attrs['fps']}) differs from specified FPS ({args.fps})")
                                logger.error("Cannot resume with different FPS. Use the same FPS as the existing file.")
                                return 1
                            if 'image_height' in f.attrs and 'image_width' in f.attrs:
                                existing_height = f.attrs['image_height']
                                existing_width = f.attrs['image_width']
                                if existing_height != args.image_size[0] or existing_width != args.image_size[1]:
                                    logger.error(
                                        f"Existing image size ({existing_height}x{existing_width}) "
                                        f"differs from specified size ({args.image_size[0]}x{args.image_size[1]})"
                                    )
                                    logger.error("Cannot resume with different image size. Use the same image size as the existing file.")
                                    return 1
                            # Get next episode index
                            if 'data' in f:
                                existing_demos = [k for k in f['data'].keys() if k.startswith('demo_')]
                                if existing_demos:
                                    demo_nums = [int(k.split('_')[1]) for k in existing_demos]
                                    current_episode = max(demo_nums) + 1
                                    if start_episode == 0:
                                        start_episode = current_episode
                                    logger.info(f"Found {len(existing_demos)} existing episode(s). Will record starting from episode {current_episode + 1}")
                        resumed_files.add(str(output_path))
                    
                    # Save episode
                    with h5py.File(output_path, 'a') as h5_file:
                        save_episode_to_h5(h5_file, episode_data, current_episode)
                    
                    logger.info(f"✓ Episode {current_episode + 1} saved! ({current_episode - start_episode + 1}/{args.num_episodes})")
                    current_episode += 1
                    
                    if (current_episode - start_episode) < args.num_episodes:
                        print(f"\nReset the environment for next episode.")
                        print(f"Press Button A('3') when ready to record episode {current_episode + 1}")
                        print("Or press Ctrl+C to finish recording.\n")
                else:
                    logger.info("Episode rejected - not saved to file")
                    print(f"\nPress Button A('3') to start a fresh recording of episode {current_episode + 1}")
                    print("Or press Ctrl+C to finish recording.\n")
                    
            except KeyboardInterrupt:
                print("\n\n⚠ Stopping recording...")
                break
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Recording complete! {current_episode - start_episode} episode(s) saved in this session.")
        logger.info(f"Total episodes in file: {current_episode}")
        logger.info(f"{'='*70}\n")
        
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
