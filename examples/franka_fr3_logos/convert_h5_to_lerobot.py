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
Convert H5 dataset to LeRobot format for Franka FR3 robot.

This script converts an H5 dataset containing robot demonstrations
to LeRobot format suitable for training policies.

NOTE: This conversion assumes quasistatic motion where the robot closely tracks 
commanded positions. Actions are derived from the next observation state, which 
approximates the commanded target position. This is valid for position-controlled 
robots with good tracking performance but may not capture dynamics, delays, or 
tracking errors present in the original commands.

Usage:
    python convert_h5_to_lerobot.py --h5_path path/to/dataset.h5 --output_dir path/to/output --repo_id my_dataset
    For EE space conversion add these args: --use_ee --urdf_path ../../src/lerobot/robots/franka_fr3/franka_fr3.urdf --ee_frame_name fr3_hand_tcp
    
    Language annotations are automatically loaded from a .txt file with the same name as the h5 file
    For example, if h5_path is "dataset.h5", it will look for "dataset.txt" in the same directory
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.franka_fr3.config_franka_fr3 import FrankaFR3Config
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.rotation import Rotation


def create_robot_config(use_ee: bool = False, image_height: int = 224, image_width: int = 224, fps: int = 30):
    """Create robot configuration with camera settings matching H5 data.
    
    Args:
        use_ee: Whether to use end-effector space
        image_height: Height of camera images
        image_width: Width of camera images
        fps: Frames per second
    """
    # Camera configuration matching H5 data dimensions
    camera_configs = {
        "front_img": RealSenseCameraConfig(
            serial_number_or_name="938422074102",
            width=image_width,
            height=image_height,
            fps=fps,
        ),
        "wrist_img": RealSenseCameraConfig(
            serial_number_or_name="919122070360",
            width=image_width,
            height=image_height,
            fps=fps,
        ),
    }
    
    return FrankaFR3Config(
        id="franka_fr3",
        cameras=camera_configs,
        use_ee=use_ee,
    )


def extract_joint_positions_from_h5(joint_states):
    """
    Extract joint positions and gripper state from the H5 joint_states data.
    
    Args:
        joint_states: Array of shape (timesteps, 3, N) where N is 8 or 9
                     containing [position, velocity, acceleration]
        
    Returns:
        Array of shape (timesteps, 8) containing 7 joint positions + gripper value
    """
    # Extract position data (first row) - use first 8 joints (7 arm + 1 gripper)
    # This handles both (T, 3, 8) and (T, 3, 9) formats - the 9th joint is unused if present
    joint_positions = joint_states[:, 0, :8]  # Shape: (timesteps, 8)
    
    return joint_positions  # Shape: (timesteps, 8)


def convert_joints_to_ee(joint_positions: np.ndarray, kinematics: RobotKinematics) -> np.ndarray:
    """
    Convert joint positions to end-effector poses using forward kinematics.
    
    Args:
        joint_positions: Array of shape (timesteps, 8) containing 7 joint positions + gripper
        kinematics: RobotKinematics instance for FK computation
        
    Returns:
        Array of shape (timesteps, 7) containing [x, y, z, wx, wy, wz, gripper]
    """
    num_timesteps = len(joint_positions)
    ee_poses = np.zeros((num_timesteps, 7), dtype=np.float32)
    
    for i in range(num_timesteps):
        # Extract joint positions (first 7 values, excluding gripper)
        joints = joint_positions[i, :7].astype(float)
        
        # Compute forward kinematics -> returns 4x4 transformation matrix
        # Note: RobotKinematics expects joint positions in degrees
        ee_transform = kinematics.forward_kinematics(np.rad2deg(joints))
        
        # Extract position (x, y, z) from translation part
        ee_poses[i, 0:3] = ee_transform[:3, 3]
        
        # Extract orientation as rotation vector (wx, wy, wz)
        rotation_matrix = ee_transform[:3, :3]
        rotation = Rotation.from_matrix(rotation_matrix)
        ee_poses[i, 3:6] = rotation.as_rotvec()
        
        # Preserve gripper value
        ee_poses[i, 6] = joint_positions[i, 7]
    
    return ee_poses


def convert_h5_to_lerobot(
    h5_path: str,
    output_dir: str,
    repo_id: str,
    fps: int = 10,
    use_ee: bool = False,
    urdf_path: str | None = None,
    ee_frame_name: str = "fr3_hand_tcp",
    joint_names: list[str] | None = None,
    task_name: str | None = None,
    skip_demos: list[int] | None = None,
    down_sample_factor: int = 1,
    logger: logging.Logger | None = None,
):
    """
    Convert H5 dataset to LeRobot format.
    
    Args:
        h5_path: Path to the H5 dataset file
        output_dir: Directory to save the converted dataset
        repo_id: Repository ID for the dataset
        fps: Frames per second for the dataset
        use_ee: Whether to convert to end-effector space (default: False for joint space)
        urdf_path: Path to robot URDF file (required if use_ee=True)
        ee_frame_name: Name of the end-effector frame in URDF (default: "fr3_hand_tcp")
        joint_names: List of joint names for FK (default: None, uses all joints)
        task_name: Default task name if language file is not found (default: None, uses h5 filename)
        skip_demos: List of demo indices (0-indexed sequential position) to skip during conversion (default: None, skips empty demos)
        down_sample_factor: Factor by which to downsample the data (default: 1, no downsampling)
        logger: Logger instance for logging messages (default: None, creates a new logger)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Converting H5 dataset from {h5_path} to LeRobot format")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Repository ID: {repo_id}")
    logger.info(f"Use end-effector space: {use_ee}")
    logger.info(f"Downsampling factor: {down_sample_factor}")
    
    # Load metadata from H5 file (fps and image size)
    with h5py.File(h5_path, 'r') as f:
        # Extract metadata if available
        if 'fps' in f.attrs:
            fps_from_metadata = int(f.attrs['fps'])
            logger.info(f"Using FPS from H5 metadata: {fps_from_metadata}")
            fps = fps_from_metadata
        else:
            logger.warning(f"No FPS metadata found in H5 file, using provided fps: {fps}")
        
        # Validate that fps is divisible by down_sample_factor
        if fps % down_sample_factor != 0:
            raise ValueError(
                f"FPS ({fps}) must be divisible by down_sample_factor ({down_sample_factor}). "
                f"Resulting FPS would be {fps / down_sample_factor}, which is not an integer."
            )
        
        # Calculate final FPS after downsampling
        final_fps = fps // down_sample_factor
        logger.info(f"Dataset FPS after downsampling: {final_fps}")
        
        if 'image_height' in f.attrs and 'image_width' in f.attrs:
            image_height = int(f.attrs['image_height'])
            image_width = int(f.attrs['image_width'])
            logger.info(f"Using image size from H5 metadata: {image_height}x{image_width}")
        else:
            # Fallback: try to infer from first episode
            logger.warning("No image size metadata found in H5 file, attempting to infer from data")
            try:
                first_demo = f['data'][next(key for key in f['data'].keys() if key.startswith('demo_'))]
                first_img = first_demo['front_img'][0]
                image_height, image_width = first_img.shape[:2]
                logger.info(f"Inferred image size from data: {image_height}x{image_width}")
            except:
                image_height, image_width = 224, 224
                logger.warning(f"Could not infer image size, using default: {image_height}x{image_width}")
    
    # Automatically look for language descriptions file with same name as h5 file
    language_descriptions = None
    h5_file_path = Path(h5_path)
    language_file_path = h5_file_path.with_suffix('.txt')
    
    # Use h5 filename (without extension) as default task name if not provided
    if task_name is None:
        task_name = h5_file_path.stem
    
    if language_file_path.exists():
        logger.info(f"Found language file: {language_file_path}")
        with open(language_file_path, 'r') as f:
            language_descriptions = [line.strip() for line in f if line.strip()]
        
        if len(language_descriptions) == 0:
            logger.warning(f"Language file {language_file_path} is empty. Using default task name: '{task_name}'")
        else:
            logger.info(f"Loaded {len(language_descriptions)} language descriptions from {language_file_path}")
            logger.info(f"Language descriptions: {language_descriptions}")
    else:
        logger.info(f"No language file found at {language_file_path}. Using default task name: '{task_name}'")
    
    # Initialize kinematics if using EE space
    kinematics = None
    if use_ee:
        if urdf_path is None:
            raise ValueError("--urdf_path is required when --use_ee is True")
        
        logger.info(f"Initializing kinematics with URDF: {urdf_path}")
        logger.info(f"End-effector frame: {ee_frame_name}")
        
        if joint_names is None:
            # Default joint names for Franka FR3
            joint_names = [
                "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
                "fr3_joint5", "fr3_joint6", "fr3_joint7"
            ]
        logger.info(f"Joint names: {joint_names}")
        
        kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=ee_frame_name,
            joint_names=joint_names,
        )
    
    # Create robot configuration to get features dynamically
    robot_config = create_robot_config(use_ee=use_ee, image_height=image_height, image_width=image_width, fps=final_fps)
    robot = make_robot_from_config(robot_config)
    
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
    
    # Add command feature to observations (commanded positions)
    # When use_ee=True, command is in EE space; otherwise in joint space
    command_names = robot_config.ee_names if use_ee else robot_config.joint_names
    command_feature = {
        "observation.command": {
            "dtype": "float32",
            "shape": (len(command_names),),
            "names": command_names,
        }
    }
    
    features = {**action_features, **obs_features, **command_feature}
    
    logger.info(f"Using {'end-effector' if use_ee else 'joint'} space features")
    logger.info(f"Action features: {list(action_features.keys())}")
    logger.info(f"Observation features: {list(obs_features.keys())}")
    logger.info(f"Command feature: observation.command with shape {command_feature['observation.command']['shape']}")
    
    # Create LeRobot dataset with video support
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=final_fps,
        root=output_dir,
        robot_type="franka_fr3",
        features=features,
        use_videos=True, 
        image_writer_processes=0,
        image_writer_threads=4,  # Use 4 threads for encoding
    )
    
    # Load H5 data and convert with video encoding
    with VideoEncodingManager(dataset):
        with h5py.File(h5_path, 'r') as f:
            data_group = f['data']
            demo_keys = [key for key in data_group.keys() if key.startswith('demo_')]
            demo_keys.sort(key=lambda x: int(x.split('_')[1]))  # Sort numerically
            
            logger.info(f"Found {len(demo_keys)} demonstrations")
            
            for episode_idx, demo_key in enumerate(demo_keys):
                logger.info(f"Processing episode {episode_idx}: {demo_key}")
                
                demo_data = data_group[demo_key]
                
                # Skip demonstrations if specified in skip_demos list (using sequential episode_idx)
                if skip_demos is not None and episode_idx in skip_demos:
                    logger.info(f"Skipping {demo_key} (episode index {episode_idx}): specified in skip_demos list")
                    continue
                
                # Extract data
                front_imgs = np.array(demo_data['front_img'])
                wrist_imgs = np.array(demo_data['wrist_img'])
                joint_states = np.array(demo_data['joint_states'])
                
                # Apply downsampling if needed
                if down_sample_factor > 1:
                    total_len = len(front_imgs)
                    indices = np.linspace(0, total_len - 1, total_len // down_sample_factor).astype(int)
                    logger.info(f"Downsampling from {total_len} frames to {len(indices)} frames (factor: {down_sample_factor})")
                    
                    front_imgs = front_imgs[indices]
                    wrist_imgs = wrist_imgs[indices]
                    joint_states = joint_states[indices]
                
                # Extract joint positions and gripper state (now combined)
                observation_state = extract_joint_positions_from_h5(joint_states)
                
                # Load command data if available, otherwise create empty arrays
                # Command from H5 has 7 values (arm joints), we need to add gripper from observations
                if 'command' in demo_data:
                    commands_7dof = np.array(demo_data['command'])  # Shape: (T, 7)
                    
                    # Apply downsampling to commands if needed
                    if down_sample_factor > 1:
                        commands_7dof = commands_7dof[indices]
                    
                    # Append gripper values from observation_state to make it (T, 8)
                    gripper_values = observation_state[:, -1:]  # Shape: (T, 1) - last column is gripper
                    commands = np.concatenate([commands_7dof, gripper_values], axis=1)  # Shape: (T, 8)
                else:
                    logger.warning(f"No 'command' data found in {demo_key}, using observation state as command")
                    commands = observation_state.copy().astype(np.float32)  # Use full observation state (T, 8)
                
                # Convert to EE space if requested
                if use_ee:
                    logger.info(f"Converting joint positions to EE poses for episode {episode_idx}")
                    observation_state = convert_joints_to_ee(observation_state, kinematics)
                    commands = convert_joints_to_ee(commands, kinematics)
                
                episode_length = len(front_imgs)
                logger.info(f"Episode {episode_idx} has {episode_length} frames")
                
                # Select task description for this episode (rotate through descriptions sequentially)
                if language_descriptions is not None:
                    episode_task = language_descriptions[episode_idx % len(language_descriptions)]
                    logger.info(f"Episode {episode_idx} task: '{episode_task}'")
                else:
                    episode_task = task_name
                
                # Process each frame in the episode
                for frame_idx in range(episode_length):
                    # Prepare observation values
                    state_names = robot_config.ee_names if use_ee else robot_config.joint_names
                    # Values dict keys must match the stripped names (without .pos suffix)
                    obs_values = {name: float(observation_state[frame_idx][i]) for i, name in enumerate(state_names)}
                    obs_values.update({"front_img": front_imgs[frame_idx], "wrist_img": wrist_imgs[frame_idx]})
                    
                    # Prepare action values (next state as target)
                    next_state = observation_state[frame_idx + 1] if frame_idx < episode_length - 1 else observation_state[frame_idx]
                    action_values = {name: float(next_state[i]) for i, name in enumerate(state_names)}
                    
                    # Build frame using build_dataset_frame
                    obs_frame = build_dataset_frame(features, obs_values, OBS_STR)
                    action_frame = build_dataset_frame(features, action_values, ACTION)
                    
                    # Add command to observation frame directly (after build_dataset_frame)
                    obs_frame["observation.command"] = commands[frame_idx]
                    
                    frame_data = {**obs_frame, **action_frame, "task": episode_task}
                    
                    # Add frame to episode buffer
                    dataset.add_frame(frame_data)
                
                # Save the episode
                dataset.save_episode()
                logger.info(f"Saved episode {episode_idx}")
    
    logger.info("Dataset conversion completed!")
    logger.info(f"Total episodes: {dataset.meta.total_episodes}")
    logger.info(f"Total frames: {dataset.meta.total_frames}")
    
    return dataset


def main():
    logger = logging.getLogger("convert_h5_to_lerobot_dataset")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    logger.propagate = False

    parser = argparse.ArgumentParser(description="Convert H5 dataset to LeRobot format")
    parser.add_argument(
        "--h5_path",
        type=str,
        default="/home/local/ASUAD/prath4/Downloads/codig_robot_datasets/softtoy_in_drawer.h5",
        help="Path to the H5 dataset file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets/franka_fr3_softtoy_in_drawer",
        help="Directory to save the converted dataset"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="franka_fr3/softtoy_in_drawer",
        help="Repository ID for the dataset"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the dataset"
    )
    parser.add_argument(
        "--use_ee",
        action="store_true",
        help="Convert to end-effector space instead of joint space"
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        default=None,
        help="Path to robot URDF file (required if --use_ee is True)"
    )
    parser.add_argument(
        "--ee_frame_name",
        type=str,
        default="fr3_hand_tcp",
        help="Name of the end-effector frame in URDF (default: fr3_hand_tcp)"
    )
    parser.add_argument(
        "--joint_names",
        type=str,
        nargs="+",
        default=None,
        help="List of joint names for FK computation (default: panda_joint1-7)"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Default task name to use if language file (.txt) is not found. "
             "If not provided, uses the h5 filename (without extension). "
             "The script automatically looks for a .txt file with the same name as the h5 file."
    )
    parser.add_argument(
        "--skip_demos",
        type=int,
        nargs="*",
        default=None,
        help="List of demo indices (0-indexed sequential position in sorted list) to skip during conversion. "
             "Example: --skip_demos 0 2 5 will skip the 1st, 3rd, and 6th demos in order. "
             "If not provided, empty demos will only be warned about but not skipped."
    )
    parser.add_argument(
        "--down_sample_factor",
        type=int,
        default=1,
        help="Factor by which to downsample the data (default: 1, no downsampling). "
             "The original FPS must be divisible by this factor. "
             "Final dataset FPS will be original_fps / down_sample_factor."
    )
    
    args = parser.parse_args()
    
    # Ensure paths exist
    h5_path = Path(args.h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    # Validate EE-related arguments
    if args.use_ee and args.urdf_path is None:
        raise ValueError("--urdf_path is required when --use_ee is True")
    
    if args.use_ee and args.urdf_path:
        urdf_path = Path(args.urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    
    # Convert dataset
    convert_h5_to_lerobot(
        h5_path=str(h5_path),
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        use_ee=args.use_ee,
        urdf_path=args.urdf_path,
        ee_frame_name=args.ee_frame_name,
        joint_names=args.joint_names,
        task_name=args.task_name,
        skip_demos=args.skip_demos,
        down_sample_factor=args.down_sample_factor,
        logger=logger,
    )


if __name__ == "__main__":
    main()