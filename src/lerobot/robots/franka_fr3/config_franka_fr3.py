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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("franka_fr3")
@dataclass
class FrankaFR3Config(RobotConfig):
    # Joint names for Franka FR3 (7-DOF arm + gripper)
    joint_names: list[str] = field(default_factory=lambda: [
        "joint1", "joint2", "joint3", "joint4", 
        "joint5", "joint6", "joint7", "gripper"
    ])
    
    # End-effector space configuration
    use_ee: bool = False  # Whether to use end-effector space instead of joint space
    # EE state names (6-DOF axis angle + gripper)
    ee_names: list[str] = field(default_factory=lambda: [
        "x", "y", "z", "wx", "wy", "wz", "gripper"
    ])
    
    # ROS2 topic and action names
    joint_trajectory_topic: str = "/fr3_arm_controller/joint_trajectory"
    joint_state_topic: str = "/joint_states"
    gripper_action_name: str = "/franka_gripper/grasp"
    
    # FrankaInterface control parameters
    # alpha: Filter coefficient for position smoothing (0-1). Higher values = more aggressive tracking.
    alpha: float = 0.95
    # dt: Time step for trajectory execution (seconds)
    dt: float = 0.1
    # numb_duration: Debounce duration to prevent rapid gripper toggling (seconds)
    numb_duration: float = 2.0
    # grasp_threshold: Tuple of (close_threshold, open_threshold) for gripper width hysteresis
    # grasp_threshold: tuple[float, float] = (0.0395, 0.005) # ideal replay
    grasp_threshold: tuple[float, float] = (0.039, 0.01) # tuned
    # grasp_threshold: tuple[float, float] = (0.039, 0.038) # slacked
    
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps joint
    # names to the max_relative_target value for that joint.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    