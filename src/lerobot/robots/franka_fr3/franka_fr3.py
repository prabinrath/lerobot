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

import logging
from functools import cached_property
from typing import Any, Optional

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_franka_fr3 import FrankaFR3Config
from .franka_interface import initialize_franka_interface, FrankaInterface

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
        
        # Initialize camera systems
        self.cameras = make_cameras_from_configs(config.cameras)
        
        # ROS2 interface (initialized on connect)
        self.franka_interface: Optional[FrankaInterface] = None
        
    @property
    def _motors_ft(self) -> dict[str, type]:
        """Feature types for motor positions"""
        return {f"{joint}.pos": float for joint in self.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Feature types for cameras"""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) 
            for cam in self.cameras
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
            Dictionary containing joint positions and camera images
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        observation = {}
        
        # Get joint positions from ROS2 interface
        joint_positions = self.franka_interface.get_joint_positions()
        if joint_positions is None:
            raise RuntimeError("No joint positions available from Franka interface")
        
        for i, joint in enumerate(self.joint_names):
            observation[f"{joint}.pos"] = float(joint_positions[i])
        
        # Get camera images
        for cam_name, cam in self.cameras.items():
            observation[cam_name] = cam.capture()
            
        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send action to the robot.
        
        Args:
            action: Dictionary containing target joint positions
            
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
        
        # Return the actual action sent
        sent_action = {}
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
