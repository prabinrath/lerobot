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
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

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
        
        # Joint names for Franka FR3 (7-DOF arm + gripper)
        self.joint_names = [
            "joint1", "joint2", "joint3", "joint4", 
            "joint5", "joint6", "joint7", "gripper"
        ]
        
        # Initialize camera systems
        self.cameras = make_cameras_from_configs(config.cameras)
        
        # Robot state storage
        self._current_joint_positions = np.zeros(8)  # 7 joints + gripper
        self._current_joint_velocities = np.zeros(8)
        
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
        return self._is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the Franka FR3 robot.
        
        Args:
            calibrate: Whether to calibrate after connecting (not needed for Franka)
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(f"Connecting to Franka FR3 at {self.config.robot_ip}")
        
        try:
            # Note: In a real implementation, you would initialize the Franka control interface here
            # For now, we'll simulate the connection
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
            
        logger.info("Configuring Franka FR3 settings")
        # In a real implementation, this would set up:
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
        
        # Get joint positions
        for i, joint in enumerate(self.joint_names):
            observation[f"{joint}.pos"] = float(self._current_joint_positions[i])
        
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

        # Extract target positions
        target_positions = np.zeros(8)
        for i, joint in enumerate(self.joint_names):
            key = f"{joint}.pos"
            if key in action:
                target_positions[i] = action[key]
        
        # Apply safety limits if configured
        if self.config.max_relative_target is not None:
            if isinstance(self.config.max_relative_target, float):
                # Global limit for all joints
                max_delta = self.config.max_relative_target
                delta = target_positions - self._current_joint_positions
                delta = np.clip(delta, -max_delta, max_delta)
                target_positions = self._current_joint_positions + delta
            elif isinstance(self.config.max_relative_target, dict):
                # Per-joint limits
                for i, joint in enumerate(self.joint_names):
                    if joint in self.config.max_relative_target:
                        max_delta = self.config.max_relative_target[joint]
                        delta = target_positions[i] - self._current_joint_positions[i]
                        delta = np.clip(delta, -max_delta, max_delta)
                        target_positions[i] = self._current_joint_positions[i] + delta

        # In a real implementation, send commands to robot here
        # For simulation, just update the current position
        self._current_joint_positions = target_positions
        
        # Return the actual action sent
        sent_action = {}
        for i, joint in enumerate(self.joint_names):
            sent_action[f"{joint}.pos"] = float(target_positions[i])
            
        return sent_action

    def disconnect(self) -> None:
        """Disconnect from the robot and cleanup"""
        if not self.is_connected:
            return
            
        logger.info("Disconnecting from Franka FR3")
        
        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        # In a real implementation, properly close robot connection here
        self._is_connected = False
        
        if self.config.disable_torque_on_disconnect:
            logger.info("Torque disabled on disconnect")