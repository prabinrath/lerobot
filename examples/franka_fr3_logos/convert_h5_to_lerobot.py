#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Convert H5 dataset to LeRobot format for Franka FR3 robot.

This script converts an H5 dataset containing robot demonstrations
to LeRobot format suitable for training policies.

Usage:
    python convert_h5_to_lerobot.py --h5_path path/to/dataset.h5 --output_dir path/to/output --repo_id my_dataset
    
    # For EE space conversion:
    python convert_h5_to_lerobot.py --h5_path path/to/dataset.h5 --output_dir path/to/output --repo_id my_dataset \
        --use_ee --urdf_path ../../src/lerobot/robots/franka_fr3/franka_fr3.urdf --ee_frame_name fr3_hand_tcp
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.rotation import Rotation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def define_franka_fr3_features():
    """Define the features for Franka FR3 robot dataset (joint space)."""
    return {
        # Robot state observations (7 joint positions + gripper state)
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "axes": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "gripper"],
            },
        },
        # Camera observations
        "observation.images.front_img": {
            "dtype": "image",
            "shape": (96, 96, 3),
            "names": ["height", "width", "channels"],  # Use list format for backward compatibility
        },
        "observation.images.wrist_img": {
            "dtype": "image", 
            "shape": (96, 96, 3),
            "names": ["height", "width", "channels"],  # Use list format for backward compatibility
        },
        # Actions (7 joint targets + gripper target)
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "axes": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "gripper"],
            },
        },
    }


def define_franka_fr3_ee_features():
    """Define the features for Franka FR3 robot dataset (end-effector space)."""
    return {
        # Robot state observations (EE pose: x, y, z, wx, wy, wz + gripper state)
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "axes": ["x", "y", "z", "wx", "wy", "wz", "gripper"],
            },
        },
        # Camera observations
        "observation.images.front_img": {
            "dtype": "image",
            "shape": (96, 96, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist_img": {
            "dtype": "image", 
            "shape": (96, 96, 3),
            "names": ["height", "width", "channels"],
        },
        # Actions (EE pose targets: x, y, z, wx, wy, wz + gripper target)
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "axes": ["x", "y", "z", "wx", "wy", "wz", "gripper"],
            },
        },
    }


def extract_joint_positions_from_h5(joint_states):
    """
    Extract joint positions and gripper state from the H5 joint_states data.
    
    Args:
        joint_states: Array of shape (timesteps, 3, 9) containing [position, velocity, acceleration]
        
    Returns:
        Array of shape (timesteps, 8) containing 7 joint positions + gripper value
    """
    # Extract position data (first row) and first 7 joints + gripper (joint 7)
    joint_positions = joint_states[:, 0, :7]  # Shape: (timesteps, 7)
    gripper_values = joint_states[:, 0, 7:8]  # Shape: (timesteps, 1) - 8th joint as gripper
    
    # Combine joint positions and gripper values
    return np.column_stack([joint_positions, gripper_values])  # Shape: (timesteps, 8)


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
        joints = joint_positions[i, :7]
        
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
    """
    logger.info(f"Converting H5 dataset from {h5_path} to LeRobot format")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Repository ID: {repo_id}")
    logger.info(f"Use end-effector space: {use_ee}")
    
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
    
    # Define features based on space (joint or EE)
    if use_ee:
        features = define_franka_fr3_ee_features()
        logger.info("Using end-effector space features")
    else:
        features = define_franka_fr3_features()
        logger.info("Using joint space features")
    
    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_dir,
        robot_type="franka_fr3",
        features=features,
        use_videos=False,  # We'll use individual images
    )
    
    # Load H5 data
    with h5py.File(h5_path, 'r') as f:
        data_group = f['data']
        demo_keys = [key for key in data_group.keys() if key.startswith('demo_')]
        demo_keys.sort(key=lambda x: int(x.split('_')[1]))  # Sort numerically
        
        logger.info(f"Found {len(demo_keys)} demonstrations")
        
        for episode_idx, demo_key in enumerate(demo_keys):
            logger.info(f"Processing episode {episode_idx}: {demo_key}")
            
            demo_data = data_group[demo_key]
            
            # Extract data
            front_imgs = np.array(demo_data['front_img'])
            wrist_imgs = np.array(demo_data['wrist_img'])
            joint_states = np.array(demo_data['joint_states'])
            
            # Extract joint positions and gripper state (now combined)
            observation_state = extract_joint_positions_from_h5(joint_states)
            
            # Convert to EE space if requested
            if use_ee:
                logger.info(f"Converting joint positions to EE poses for episode {episode_idx}")
                observation_state = convert_joints_to_ee(observation_state, kinematics)
            
            episode_length = len(front_imgs)
            logger.info(f"Episode {episode_idx} has {episode_length} frames")
            
            # Process each frame in the episode
            for frame_idx in range(episode_length):
                # Prepare frame data (don't include episode_index, frame_index, timestamp as they're auto-added)
                frame_data = {
                    "task": "softtoy_in_drawer",
                    "observation.state": observation_state[frame_idx].astype(np.float32),
                    "observation.images.front_img": front_imgs[frame_idx],
                    "observation.images.wrist_img": wrist_imgs[frame_idx],
                }
                
                # For actions, we use the next state as target (simple approach)
                # For the last frame, we use the current state
                if frame_idx < episode_length - 1:
                    action = observation_state[frame_idx + 1].astype(np.float32)
                else:
                    action = observation_state[frame_idx].astype(np.float32)
                
                frame_data["action"] = action
                
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
    )


if __name__ == "__main__":
    main()