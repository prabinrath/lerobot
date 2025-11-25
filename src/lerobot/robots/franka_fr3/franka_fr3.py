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

import logging
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.rotation import Rotation

from ..robot import Robot
from .config_franka_fr3 import FrankaFR3Config

logger = logging.getLogger(__name__)


class FrankaFR3(Robot):
    """
    Franka Emika FR3 Robot implementation for LeRobot.
    
    This robot implementation follows the standard LeRobot robot interface
    and provides integration for the Franka FR3 7-DOF robotic arm with gripper.
    """

    config_class = FrankaFR3Config
    name = "franka_fr3"

    def __init__(self, config: FrankaFR3Config):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._is_calibrated = True  # Franka robots are self-calibrating
        
        # Joint names from config
        self.joint_names = config.joint_names
        
        # Camera systems (initialized on connect)
        self.cameras = None
        
        # ROS2 interface (initialized on connect)
        self.franka_interface = None
        
        # Kinematics for end-effector control (initialized if use_ee=True)
        self.kinematics = None
        if config.use_ee:
            # Get URDF path from robot package directory
            urdf_path = Path(__file__).parent / "franka_fr3_kinematics.urdf"
            # Joint names for FK/IK (excluding gripper)
            fk_joint_names = [
                "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
                "fr3_joint5", "fr3_joint6", "fr3_joint7"
            ]
            self.kinematics = RobotKinematics(
                urdf_path=str(urdf_path),
                target_frame_name="fr3_hand_tcp",
                joint_names=fk_joint_names,
            )
            logger.info(f"Initialized kinematics for end-effector control with URDF: {urdf_path}")
        
    @property
    def _motors_ft(self) -> dict[str, type]:
        """Feature types for motor positions"""
        if self.config.use_ee:
            # End-effector space
            return {f"{name}.pos": float for name in self.config.ee_names}
        else:
            # Joint space
            return {f"{joint}.pos": float for joint in self.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Feature types for cameras"""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) 
            for cam in self.config.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """
        Observation features combining joint positions and camera feeds.
        Returns dictionary mapping feature names to their types/shapes.
        """
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property  
    def action_features(self) -> dict[str, type]:
        """
        Action features for robot control.
        Returns dictionary mapping action names to their types.
        """
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected"""
        return (
            self._is_connected 
            and self.franka_interface is not None
            and self.franka_interface.is_ready()
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = False) -> None:
        """
        Connect to the Franka FR3 robot.
        
        Args:
            calibrate: Whether to calibrate after connecting (not needed for Franka)
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        logger.info("Connecting to Franka FR3")
        
        from lerobot.cameras.utils import make_cameras_from_configs
        from .franka_interface import initialize_franka_interface

        self.cameras = make_cameras_from_configs(self.config.cameras)
        
        try:
            # Initialize ROS2 interface
            self.franka_interface = initialize_franka_interface(
                node_name=f"franka_fr3_{id(self)}",
                joint_trajectory_topic=self.config.joint_trajectory_topic,
                joint_state_topic=self.config.joint_state_topic,
                gripper_action_name=self.config.gripper_action_name,
                alpha=self.config.alpha,
                dt=self.config.dt,
                numb_duration=self.config.numb_duration,
                grasp_threshold=self.config.grasp_threshold,
            )
            
            self._is_connected = True
            logger.info("Successfully connected to Franka FR3")
            
            # Connect cameras
            for cam in self.cameras.values():
                cam.connect()
                
        except Exception as e:
            logger.error(f"Failed to connect to Franka FR3: {e}")
            raise

    @property
    def is_calibrated(self) -> bool:
        """Franka robots are self-calibrating"""
        return self._is_calibrated

    def calibrate(self) -> None:
        """
        Calibrate the robot. Franka robots are self-calibrating,
        so this is typically a no-op.
        """
        logger.info("Franka FR3 is self-calibrating - no manual calibration needed")
        self._is_calibrated = True

    def configure(self) -> None:
        """
        Configure the robot with appropriate settings.
        This includes setting control modes, safety limits, etc.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
            
        logger.info("Configured Franka FR3 settings")
        # Ideally, this would set up:
        # - Control modes (position, velocity, torque)
        # - Safety limits and collision behavior
        # - Impedance parameters
        # - etc.

    def get_observation(self) -> dict[str, Any]:
        """
        Get current observation from the robot.
        
        Returns:
            Dictionary containing joint positions (or EE pose if use_ee=True) and camera images
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        observation = {}
        
        # Get joint positions from ROS2 interface
        joint_positions = self.franka_interface.get_joint_positions()
        if joint_positions is None:
            raise RuntimeError("No joint positions available from Franka interface")
        
        # Convert to EE pose if use_ee is enabled
        if self.config.use_ee:
            # Extract arm joint positions (first 7 joints, in radians)
            arm_joints = joint_positions[:7]
            gripper_pos = joint_positions[7]
            
            # Compute forward kinematics (convert radians to degrees for RobotKinematics)
            ee_transform = self.kinematics.forward_kinematics(np.rad2deg(arm_joints))
            
            # Extract position (x, y, z)
            pos = ee_transform[:3, 3]
            
            # Extract orientation as rotation vector (wx, wy, wz) - axis-angle representation
            rotation_matrix = ee_transform[:3, :3]
            rotation = Rotation.from_matrix(rotation_matrix)
            rotvec = rotation.as_rotvec()
            
            # Build observation with EE pose
            for i, name in enumerate(self.config.ee_names[:-1]):  # x, y, z, wx, wy, wz
                if i < 3:  # position
                    observation[f"{name}.pos"] = float(pos[i])
                else:  # orientation (rotation vector)
                    observation[f"{name}.pos"] = float(rotvec[i - 3])
            
            # Add gripper
            observation[f"{self.config.ee_names[-1]}.pos"] = float(gripper_pos)
        else:
            # Joint space - use joint positions directly
            for i, joint in enumerate(self.joint_names):
                observation[f"{joint}.pos"] = float(joint_positions[i])
        
        # Get camera images
        for cam_name, cam in self.cameras.items():
            observation[cam_name] = cam.async_read()
            
        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send action to the robot.
        
        Args:
            action: Dictionary containing target joint positions (or EE pose if use_ee=True)
            
        Returns:
            Dictionary containing the actual action sent (potentially clipped)
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # Get current joint positions
        current_joint_positions = self.franka_interface.get_joint_positions()
        if current_joint_positions is None:
            logger.warning("Cannot send action: no current joint positions available")
            return {}
        
        # Extract target positions
        target_positions = np.zeros(8)
        
        if self.config.use_ee:
            # EE space - convert EE pose to joint positions using inverse kinematics
            # Extract EE pose from action
            x = action.get(f"{self.config.ee_names[0]}.pos", None)
            y = action.get(f"{self.config.ee_names[1]}.pos", None)
            z = action.get(f"{self.config.ee_names[2]}.pos", None)
            wx = action.get(f"{self.config.ee_names[3]}.pos", None)
            wy = action.get(f"{self.config.ee_names[4]}.pos", None)
            wz = action.get(f"{self.config.ee_names[5]}.pos", None)
            gripper_pos = action.get(f"{self.config.ee_names[6]}.pos", None)
            
            if None in (x, y, z, wx, wy, wz, gripper_pos):
                raise ValueError(
                    f"Missing required end-effector pose components in action. "
                    f"Expected: {[f'{name}.pos' for name in self.config.ee_names]}"
                )
            
            # Build desired 4x4 transform from position + rotation vector (axis-angle)
            t_des = np.eye(4, dtype=float)
            t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
            t_des[:3, 3] = [x, y, z]
            
            # Use current joint positions as initial guess for IK (in degrees)
            q_curr = np.rad2deg(current_joint_positions[:7])
            
            # Compute inverse kinematics (returns joint positions in degrees)
            q_target_deg = self.kinematics.inverse_kinematics(q_curr, t_des)
            
            # Convert back to radians
            target_positions[:7] = np.deg2rad(q_target_deg)
            target_positions[7] = gripper_pos
        else:
            # Joint space - extract target positions directly
            for i, joint in enumerate(self.joint_names):
                key = f"{joint}.pos"
                if key in action:
                    target_positions[i] = action[key]
                else:
                    # Keep current position if not specified
                    target_positions[i] = current_joint_positions[i]
        
        # Apply safety limits if configured (only to arm joints, not gripper)
        if self.config.max_relative_target is not None:
            if isinstance(self.config.max_relative_target, float):
                # Global limit for arm joints only (first 7)
                max_delta = self.config.max_relative_target
                delta = target_positions[:7] - current_joint_positions[:7]
                delta = np.clip(delta, -max_delta, max_delta)
                target_positions[:7] = current_joint_positions[:7] + delta
            elif isinstance(self.config.max_relative_target, dict):
                # Per-joint limits (only apply to arm joints, not gripper)
                for i, joint in enumerate(self.joint_names[:7]):  # Only first 7 joints
                    if joint in self.config.max_relative_target:
                        max_delta = self.config.max_relative_target[joint]
                        delta = target_positions[i] - current_joint_positions[i]
                        delta = np.clip(delta, -max_delta, max_delta)
                        target_positions[i] = current_joint_positions[i] + delta

        # Send arm joint positions (first 7 joints)
        arm_positions = target_positions[:7]
        self.franka_interface.send_joint_positions(arm_positions, use_filtering=True)
        
        # Send gripper position (8th position)
        gripper_position = target_positions[7]
        self.franka_interface.send_gripper_position(gripper_position)
        
        # Return the actual action sent (in the same format as the input)
        sent_action = {}
        if self.config.use_ee:
            # Convert joint positions back to EE pose for return value
            ee_transform = self.kinematics.forward_kinematics(np.rad2deg(target_positions[:7]))
            pos = ee_transform[:3, 3]
            rotvec = Rotation.from_matrix(ee_transform[:3, :3]).as_rotvec()
            
            sent_action[f"{self.config.ee_names[0]}.pos"] = float(pos[0])
            sent_action[f"{self.config.ee_names[1]}.pos"] = float(pos[1])
            sent_action[f"{self.config.ee_names[2]}.pos"] = float(pos[2])
            sent_action[f"{self.config.ee_names[3]}.pos"] = float(rotvec[0])
            sent_action[f"{self.config.ee_names[4]}.pos"] = float(rotvec[1])
            sent_action[f"{self.config.ee_names[5]}.pos"] = float(rotvec[2])
            sent_action[f"{self.config.ee_names[6]}.pos"] = float(target_positions[7])
        else:
            # Joint space - return joint positions
            for i, joint in enumerate(self.joint_names):
                sent_action[f"{joint}.pos"] = float(target_positions[i])
            
        return sent_action

    def disconnect(self) -> None:
        """Disconnect from the robot and cleanup"""
        if not self._is_connected:
            return
            
        logger.info("Disconnecting from Franka FR3")
        
        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        # Shutdown ROS2 interface
        if self.franka_interface is not None:
            self.franka_interface.shutdown()
            self.franka_interface = None
        
        self._is_connected = False
