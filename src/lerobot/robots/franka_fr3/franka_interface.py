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
FrankaInterface: ROS2-based communication layer for Franka FR3 robot.

This module provides a clean interface for ROS2 communication with the Franka FR3 robot.
The FrankaInterface class handles:
- Joint state subscriptions
- Joint position command publishing
- Gripper control via Grasp action client
- Thread-safe state management
- Hysteresis and timing-based gripper control to prevent rapid toggling

The interface is initialized automatically by the FrankaFR3 robot class and runs
ROS2 communication in a separate thread to avoid blocking the main LeRobot control loop.

Gripper Control Logic:
    The gripper uses a control scheme with:
    - Grasp action client for both opening and closing
    - Numb duration (default 2.0s) to prevent rapid state changes
    - Dual thresholds (close_threshold, open_threshold) for hysteresis
    - State-dependent timing to avoid command flooding

Example Usage:
    # Typically used internally by FrankaFR3 class
    interface = initialize_franka_interface(
        node_name="franka_interface",
        numb_duration=2.0,
        grasp_threshold=(0.039, 0.038)
    )
    
    # Get current joint positions
    positions = interface.get_joint_positions()
    
    # Send joint positions
    target = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    interface.send_joint_positions(target)
    
    # Send gripper position (with timing logic)
    interface.send_gripper_position(0.08)  # Will open if past numb_duration
    interface.send_gripper_position(0.0)   # Will close if past numb_duration
"""

import logging
import threading
from threading import Lock
from typing import Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from franka_msgs.action import Grasp
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time

logger = logging.getLogger(__name__)


class FrankaInterface(Node):
    """
    ROS2-based interface for Franka FR3 robot communication.
    
    This class handles all ROS2-related communication with the Franka FR3 robot,
    including joint state subscription, trajectory publishing, and gripper control.
    """

    def __init__(
        self,
        node_name: str = "franka_interface",
        joint_trajectory_topic: str = "/fr3_arm_controller/joint_trajectory",
        joint_state_topic: str = "/joint_states",
        gripper_action_name: str = "/franka_gripper/grasp",
        alpha: float = 0.95,
        dt: float = 0.01,
        numb_duration: float = 2.0,
        grasp_threshold: tuple = (0.039, 0.038),
    ):
        """
        Initialize the Franka ROS2 interface.
        
        Args:
            node_name: Name for the ROS2 node
            joint_trajectory_topic: Topic for publishing joint trajectories
            joint_state_topic: Topic for subscribing to joint states
            gripper_action_name: Action server name for gripper control
            alpha: Filter coefficient for position smoothing (0-1)
            dt: Time step for trajectory execution
            numb_duration: Numb duration to prevent rapid gripper toggling (seconds)
            grasp_threshold: Tuple of (close_threshold, open_threshold) for gripper width
        """
        super().__init__(node_name)
        
        self.alpha = alpha
        self.dt = dt
        self.js_mutex = Lock()
        
        # Joint state storage
        self._latest_joint_state: Optional[JointState] = None
        self._current_joint_positions: Optional[np.ndarray] = None
        
        # Gripper state and timing
        self.is_grasped = False
        self.numb_duration = numb_duration
        self.grasp_threshold = grasp_threshold
        self.gripper_close_time = self.get_clock().now()
        self.gripper_open_time = self.get_clock().now()
        
        # Joint names for Franka FR3 (7 arm joints only)
        self.joint_names = ['fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4', 
                           'fr3_joint5', 'fr3_joint6', 'fr3_joint7']
        
        # Set up ROS2 publishers and subscribers
        self.franka_cmd_pub = self.create_publisher(
            JointTrajectory, 
            joint_trajectory_topic, 
            10
        )
        
        self.joint_state_sub = self.create_subscription(
            JointState,
            joint_state_topic,
            self._franka_joint_state_callback,
            10
        )
        
        # Set up gripper action client
        self.gripper_client = ActionClient(self, Grasp, gripper_action_name)
        
        logger.info("Waiting for gripper action server...")
        self.gripper_client.wait_for_server()
        logger.info("Gripper action server connected")
        
        # Spin the node in a separate thread
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        
        # prepare for rollout
        self.grasp_open()
        time.sleep(2.0) 

        logger.info(f"FrankaInterface initialized with node name: {node_name}")

    def _spin(self):
        """Spin the ROS2 node in a separate thread."""
        try:
            self._executor.spin()
        except Exception as e:
            logger.error(f"Error in ROS2 spin thread: {e}")

    def _franka_joint_state_callback(self, msg: JointState) -> None:
        """
        Callback for joint state messages.
        
        Args:
            msg: JointState message containing current joint positions
        """
        with self.js_mutex:
            self._latest_joint_state = msg
            # Store first 7 joints (arm) + gripper position
            if len(msg.position) >= 7:
                self._current_joint_positions = np.asarray(msg.position[:8])

    def get_joint_positions(self) -> Optional[np.ndarray]:
        """
        Get the current joint positions.
        
        Returns:
            Array of 8 joint positions (7 arm joints + gripper) or None if not available
        """
        with self.js_mutex:
            if self._current_joint_positions is not None:
                return self._current_joint_positions.copy()
            return None

    def get_arm_positions(self) -> Optional[np.ndarray]:
        """
        Get the current arm joint positions (excluding gripper).
        
        Returns:
            Array of 7 arm joint positions or None if not available
        """
        joint_positions = self.get_joint_positions()
        if joint_positions is not None:
            return joint_positions[:7]
        return None

    def send_joint_positions(
        self,
        target_positions: np.ndarray,
        time_from_start: Optional[float] = None,
        use_filtering: bool = True,
    ) -> None:
        """
        Send target joint positions to the robot.
        
        Args:
            target_positions: Target positions for the 7 arm joints
            time_from_start: Time to reach target (if None, uses self.dt)
            use_filtering: Whether to apply position filtering
        """
        if len(target_positions) != 7:
            raise ValueError(f"Expected 7 joint positions, got {len(target_positions)}")
        
        current_positions = self.get_arm_positions()
        if current_positions is None:
            logger.warning("No current joint positions available, cannot send trajectory")
            return
        
        # Apply filtering if requested
        if use_filtering:
            filtered_positions = (1 - self.alpha) * current_positions + self.alpha * target_positions
        else:
            filtered_positions = target_positions
        
        # Create trajectory message
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = filtered_positions.tolist()
        
        # Set time from start
        if time_from_start is None:
            time_from_start = self.dt
        
        t = Duration(sec=0, nanosec=int(time_from_start * 1e9))
        point.time_from_start = t
        
        traj.points = [point]
        self.franka_cmd_pub.publish(traj)

    def grasp_close(self) -> None:
        """
        Close the gripper using grasp action.
        """
        grasp_goal = Grasp.Goal()
        grasp_goal.width = 0.0
        grasp_goal.epsilon.inner = 0.08
        grasp_goal.epsilon.outer = 0.08
        grasp_goal.speed = 0.1
        grasp_goal.force = 5.0
        self.gripper_client.send_goal_async(grasp_goal)
        logger.debug("Gripper grasp close command sent")

    def grasp_open(self) -> None:
        """
        Open the gripper using grasp action.
        """
        grasp_goal = Grasp.Goal()
        grasp_goal.width = 0.08
        grasp_goal.epsilon.inner = 0.08
        grasp_goal.epsilon.outer = 0.08
        grasp_goal.speed = 0.1
        grasp_goal.force = 5.0
        self.gripper_client.send_goal_async(grasp_goal)
        logger.debug("Gripper grasp open command sent")

    def send_gripper_position(self, gripper_position: float) -> None:
        """
        Send gripper position command with numb duration logic.
        
        This implements hysteresis and timing logic to prevent rapid toggling:
        - Waits for numb_duration after state changes before allowing new changes
        - Uses different thresholds for closing vs opening
        
        Args:
            gripper_position: Target gripper position from policy/action
        """
        current_time = self.get_clock().now()
        
        # Update timing when state is active
        if self.is_grasped:
            self.gripper_open_time = current_time
        
        # Check if we should close the gripper
        if not self.is_grasped and \
           (current_time - self.gripper_open_time).nanoseconds / 1e9 > self.numb_duration and \
           gripper_position < self.grasp_threshold[0]:
            self.grasp_close()
            self.is_grasped = True
            self.gripper_close_time = current_time
            logger.info(f"Gripper closed (position={gripper_position:.4f})")
        
        # Check if we should open the gripper
        if self.is_grasped and \
           (current_time - self.gripper_close_time).nanoseconds / 1e9 > self.numb_duration and \
           gripper_position > self.grasp_threshold[1]:
            self.grasp_open()
            self.is_grasped = False
            logger.info(f"Gripper opened (position={gripper_position:.4f})")

    def is_ready(self) -> bool:
        """
        Check if the interface has received joint state data.
        
        Returns:
            True if joint states are available, False otherwise
        """
        return self._current_joint_positions is not None

    def shutdown(self) -> None:
        """Shutdown the ROS2 interface and cleanup resources."""
        logger.info("Shutting down FrankaInterface")
        self._executor.shutdown()
        self.destroy_node()


def initialize_franka_interface(
    node_name: str = "franka_interface",
    joint_trajectory_topic: str = "/fr3_arm_controller/joint_trajectory",
    joint_state_topic: str = "/joint_states",
    gripper_action_name: str = "/franka_gripper/grasp",
    alpha: float = 0.95,
    dt: float = 0.1,
    numb_duration: float = 2.0,
    grasp_threshold: tuple = (0.039, 0.038),
    **kwargs
) -> FrankaInterface:
    """
    Initialize ROS2 and create a FrankaInterface instance.
    
    Args:
        node_name: Name for the ROS2 node
        joint_trajectory_topic: Topic for publishing joint trajectories
        joint_state_topic: Topic for subscribing to joint states
        gripper_action_name: Action server name for gripper control
        alpha: Filter coefficient for position smoothing (0-1)
        dt: Time step for trajectory execution
        numb_duration: Numb duration to prevent rapid gripper toggling (seconds)
        grasp_threshold: Tuple of (close_threshold, open_threshold) for gripper width
        **kwargs: Additional arguments to pass to FrankaInterface constructor
        
    Returns:
        Initialized FrankaInterface instance
    """
    if not rclpy.ok():
        rclpy.init()
    
    interface = FrankaInterface(
        node_name=node_name,
        joint_trajectory_topic=joint_trajectory_topic,
        joint_state_topic=joint_state_topic,
        gripper_action_name=gripper_action_name,
        alpha=alpha,
        dt=dt,
        numb_duration=numb_duration,
        grasp_threshold=grasp_threshold,
        **kwargs
    )
    
    # Wait for initial joint state
    logger.info("Waiting for initial joint state...")
    timeout = 5.0
    start_time = interface.get_clock().now()
    
    while not interface.is_ready():
        if (interface.get_clock().now() - start_time).nanoseconds / 1e9 > timeout:
            logger.warning("Timeout waiting for initial joint state")
            break
        rclpy.spin_once(interface, timeout_sec=0.1)
    
    if interface.is_ready():
        logger.info("FrankaInterface ready")
    
    return interface
