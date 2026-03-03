#!/usr/bin/env python3
"""
Interactive rollout server for the Franka FR3 + StageCraft setup.

Listens on /rollout_command (ROS2) and runs async inference rollouts.

Usage:
    python deploy_with_stagecraft.py [OPTIONS]

Example:
    python deploy_with_stagecraft.py \\
        --server_address 127.0.0.1:8080 \\
        --policy_device cuda \\
        --fps 10
"""

import argparse
import logging
import threading
import time
from pathlib import Path
import cv2
import rclpy
import numpy as np

from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots.franka_fr3.config_franka_fr3 import FrankaFR3Config
from vlm_sam_scoring_gemini import SAM_VLM_Planner
from pointcloud_camera import RealSensePointCloudCamera
from pick_place import PickPlaceHandler, PrimitiveParam


def main():
    logger = logging.getLogger("deploy_with_stagecraft")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Interactive rollout server for Franka FR3 + StageCraft (async inference)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080",
                        help="Policy server address (host:port)")
    # Robot
    parser.add_argument("--robot_id", type=str, default="franka_fr3",
                        help="Robot ID used for calibration files")
    parser.add_argument("--use_ee", action="store_true",
                        help="Use end-effector space instead of joint space")
    parser.add_argument("--max_relative_target", type=float, default=0.05,
                        help="Max relative joint movement per step (safety limit)")
    # Policy / control
    parser.add_argument("--policy_device", type=str, default="cuda",
                        help="Device for policy inference (cuda, cpu, mps)")
    parser.add_argument("--policy_type", type=str, default="",
                        help="Policy type (e.g. act, pi0, smolvla)")
    parser.add_argument("--checkpoint_path", type=str, default="",
                        help="Path to trained policy checkpoint")
    parser.add_argument("--task", type=str, default="",
                        help="Task description for the robot to execute")
    parser.add_argument("--actions_per_chunk", type=int, default=16,
                        help="Number of actions per inference chunk")
    parser.add_argument("--chunk_size_threshold", type=float, default=0.5,
                        help="Queue threshold for requesting new actions (0.0-1.0)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Control frequency in Hz")
    parser.add_argument("--aggregate_fn_name", type=str, default="weighted_average",
                        choices=["weighted_average", "latest_only", "average", "conservative"],
                        help="Action chunk aggregation function")
    parser.add_argument("--max_rollout_steps", type=int, default=None,
                        help="Max rollout steps per episode (None = unlimited)")
    # Debug
    parser.add_argument("--debug_visualize_queue_size", action="store_true",
                        help="Visualize action queue size after each rollout")
    parser.add_argument("--experiment_name", type=str, default="",
                        help="Experiment name; if set, videos are saved to logs/stagecraft/<experiment_name>")

    # StageCraft
    parser.add_argument("--num_incontext_rollouts", type=int, default=10,
                        help="Number of incontext VLA rollouts")
    parser.add_argument("--eval_rollouts", type=int, default=10,
                        help="Number of incontext VLA rollouts")

    args = parser.parse_args()
    args.experiment_name += f"_{args.policy_type}"

    robot_config = create_robot_config(args)

    logger.info("=" * 60)
    logger.info("StageCraft Interactive Rollout Server")
    logger.info(f"  Server:    {args.server_address}")
    logger.info(f"  Robot ID:  {args.robot_id}")
    logger.info(f"  Device:    {args.policy_device}")
    logger.info(f"  FPS:       {args.fps}")
    logger.info(f"  Max steps: {args.max_rollout_steps or 'unlimited'}")
    logger.info("=" * 60)

    rclpy.init()
    rollout_node = RolloutManager(robot_config, args, logger)
    spin_thread = threading.Thread(target=rclpy.spin, args=(rollout_node.get_node(),), daemon=True)
    spin_thread.start()

    base_video_path = Path("logs/stagecraft") / args.experiment_name

    # Run In-context rollouts
    if args.num_incontext_rollouts > 0:
        # Initialise rollout counter from already-sorted videos in success/failure.
        sorted_videos = list((base_video_path / "success").glob("*.mp4")) + list((base_video_path / "failure").glob("*.mp4"))
        used_indices = {int(p.stem.rsplit("_", 1)[-1]) for p in sorted_videos if p.stem.rsplit("_", 1)[-1].isdigit()}
        rollout_idx = max(used_indices, default=-1) + 1

        logger.info("=" * 60)
        logger.info("StageCraft: Starting in-context rollouts...")
        logger.info("=" * 60)
        for _ in range(args.num_incontext_rollouts):
            rollout_node.run_inference()
            move_rollout_videos(base_video_path, rollout_node.is_success, rollout_idx, logger)
            rollout_idx += 1

    # Update paths based on rollout results
    success_path = base_video_path / "success"
    failure_path = base_video_path / "failure"

    if rollout_node.cached_async_client is None:
        rollout_node.cached_async_client = load_robot_client(robot_config=robot_config, args=args, logger=logger)
        rollout_node.cached_async_checkpoint_path = args.checkpoint_path
    
    front_cam = rollout_node.cached_async_client.robot.cameras.get("front_img")
    wrist_cam = rollout_node.cached_async_client.robot.cameras.get("wrist_img")
    t_base_cam = RealSensePointCloudCamera.make_transform(
        1.41104,   0.155109,  1.0711,               # translation  (metres)
       -0.625165,  0.650804,  0.253996, -0.348007,  # quaternion   x y z w
    )
    pointcloud_camera = RealSensePointCloudCamera(front_cam, T_base_cam=t_base_cam)
    pickplace_node = PickPlaceHandler(pointcloud_camera)
    sam_vlm_planner = SAM_VLM_Planner()

    vlm_context = sam_vlm_planner.process_videos_with_tags(
        directories=[str(success_path), str(failure_path)],
        performance_tags=['success', 'failure'],
    )

    if args.eval_rollouts > 0:
        logger.info("=" * 60)
        logger.info("StageCraft: Starting eval rollouts...")
        logger.info("=" * 60)
    # Run with stagecraft
    for idx in range(args.eval_rollouts):
        rollout_node.wait_for_start()
        
        front_img_ = cv2.cvtColor(front_cam.async_read(), cv2.COLOR_RGB2BGR)
        wrist_img_ = cv2.cvtColor(wrist_cam.async_read(), cv2.COLOR_RGB2BGR)
        front_img = cv2.resize(front_img_, (224, 224))
        wrist_img = cv2.resize(wrist_img_, (224, 224))
        current_observation = np.hstack([front_img, wrist_img])
        throw_points = sam_vlm_planner.sam_vlm_planner(
            results=vlm_context,
            current_observation=current_observation,
            task_instruction=args.task,
            front_cam_img=cv2.cvtColor(front_img_, cv2.COLOR_BGR2RGB),
            run_folder=f"logs/stagecraft/{args.experiment_name}/eval_{idx}"
        )

        for point in throw_points:
            x, y = point
            data = pointcloud_camera.get_data()

            pt = data.point_at(x, y)
            if np.isnan(pt).any():
                print(f"pixel ({x}, {y}) → no valid depth, skipping.")
                continue
            px, py, pz = float(pt[0]), float(pt[1]), float(pt[2])

            pick_pose = PrimitiveParam(
                x=px+0.03, y=py+0.025, z=max(pz-0.02, 0.05),
                yaw=pickplace_node.pick_defaults.yaw,
                approach_height=pickplace_node.pick_defaults.approach_height,
            )
            ok = pickplace_node.send_pick(pick_pose)
            if not ok:
                print("Pick failed — aborting place.")
                return
            pickplace_node.send_place(pickplace_node.throw_place_pose)

        rollout_node.send_reset()
        rollout_node.run_inference()

    return 0


def move_rollout_videos(base_path: Path, is_success: bool | None, rollout_idx: int, logger=None) -> None:
    """Move all .mp4s from base_path into success/ or failure/, renamed with rollout_idx."""
    if is_success is None:
        if logger:
            logger.warning("move_rollout_videos: is_success is None, skipping")
        return
    dest_folder = base_path / ("success" if is_success else "failure")
    dest_folder.mkdir(parents=True, exist_ok=True)
    for video_file in sorted(base_path.glob("*.mp4")):
        cam_name = video_file.stem.rsplit("_", 1)[0]
        dest = dest_folder / f"{cam_name}_{rollout_idx:04d}.mp4"
        video_file.rename(dest)
        if logger:
            logger.info(f"Moved {video_file.name} -> {dest_folder.name}/{dest.name}")


def create_robot_config(args):
    """Build camera and robot configuration from parsed args."""
    camera_configs = {
        "front_img": RealSenseCameraConfig(
            serial_number_or_name="938422074102",
            width=640,
            height=480,
            fps=30,
            rotation=Cv2Rotation.ROTATE_180,
            use_depth=True,
        ),
        "wrist_img": RealSenseCameraConfig(
            serial_number_or_name="919122070360",
            width=640,
            height=480,
            fps=30,
        ),
    }

    robot_config = FrankaFR3Config(
        id=args.robot_id,
        cameras=camera_configs,
        dt=1 / args.fps,
        use_ee=args.use_ee,
    )
    robot_config.max_relative_target = args.max_relative_target
    return robot_config


def load_robot_client(robot_config, args, logger):
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient
    
    client_config = RobotClientConfig(
        robot=robot_config,
        server_address=args.server_address,
        policy_type=args.policy_type,
        pretrained_name_or_path=str(args.checkpoint_path),
        policy_device=args.policy_device,
        task=args.task,
        actions_per_chunk=args.actions_per_chunk,
        chunk_size_threshold=args.chunk_size_threshold,
        fps=args.fps,
        aggregate_fn_name=args.aggregate_fn_name,
        debug_visualize_queue_size=args.debug_visualize_queue_size,
        max_rollout_steps=args.max_rollout_steps,
        rollout_video_path=str(Path("logs/stagecraft") / args.experiment_name) if args.experiment_name else None,
    )
    client = RobotClient(client_config)

    logger.info("Connecting to policy server and robot...")
    if not client.start():
        logger.error("Failed to start robot client")
        return client
    logger.info("Successfully connected!")

    return client


def run_async_inference(robot_config, args, logger, stop_event=None, client=None):
    """Run with async inference (policy server)
       Note: Policies with n_obs_steps > 1 are not yet supported for Async inference
             as the policy server does not manage the observation queue properly. affected
             policies are (DP, VQ-BeT)
    
    Args:
        robot_config: Robot configuration
        args: Command line arguments
        logger: Logger instance to use
        stop_event: Optional threading.Event to signal early termination
        client: Optional pre-created RobotClient (for caching in interactive mode)
    
    Returns:
        Tuple of (return_code, client) for caching
    """
    from lerobot.async_inference.helpers import visualize_action_queue_size
    
    # Create client if not provided (non-interactive mode)
    owns_client = client is None
    if client is None:
        client = load_robot_client(robot_config=robot_config, args=args, logger=logger)    
    try:
        if owns_client:
            logger.info("Connecting to policy server and robot...")
            if not client.start():
                logger.error("Failed to start robot client")
                return client
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
                    logger.info("Stop signal received, ending rollout...")
                    client.shutdown_event.set()
                    break
                time.sleep(0.1)
        
        if stop_event:
            threading.Thread(target=stop_monitor, daemon=True).start()
        
        # Run the control loop (blocking)
        client.control_loop(args.task)
        
        # Stop the threads but keep connection alive for reuse
        logger.info("Control loop completed.")
        client.shutdown_event.set()
        action_receiver_thread.join(timeout=5.0)
        
        # Optionally visualize queue size statistics
        if args.debug_visualize_queue_size and hasattr(client, 'action_queue_size'):
            logger.info("Displaying action queue size visualization...")
            visualize_action_queue_size(client.action_queue_size)
        
    except KeyboardInterrupt:
        logger.info("Stopping robot client...")
        client.stop()
        action_receiver_thread.join(timeout=5.0)
        
        if args.debug_visualize_queue_size and hasattr(client, 'action_queue_size'):
            logger.info("Displaying action queue size visualization...")
            visualize_action_queue_size(client.action_queue_size)
        
    except Exception as e:
        logger.error(f"Robot client error: {e}")
        return client
    
    return client


class RolloutManager:
    """Manages rollouts: handles spacemouse input and runs async inference.

    Must be instantiated after rclpy.init().
    Call on_command(...) to start a rollout.
    """

    def __init__(self, robot_config, args, logger):
        import rclpy
        from std_msgs.msg import Empty
        from sensor_msgs.msg import Joy

        self._node = rclpy.create_node("rollout_manager")
        self._node.create_subscription(Joy, "/spacenav/joy", self._joy_callback, 10)
        self._reset_pub = self._node.create_publisher(Empty, "/franka_reset", 10)

        self.robot_config = robot_config
        self.args = args
        self.logger = logger

        self.running = False
        self.awaiting_result = False
        self.awaiting_start = False
        self.is_success = None
        self.stop_event = threading.Event()
        self.start_event = threading.Event()

        self.cached_async_checkpoint_path = None
        self.cached_async_client = None

    def get_node(self):
        """Return the underlying rclpy node (for spinning)."""
        return self._node

    def send_reset(self) -> None:
        """Publish a robot reset command on /franka_reset."""
        from std_msgs.msg import Empty
        self._reset_pub.publish(Empty())
        self.logger.info("Robot reset command sent... Will wait for 3 secs")
        time.sleep(3)

    def wait_for_start(self) -> None:
        """Block until the spacemouse button T is pressed."""
        self.logger.info("Waiting for spacemouse button T to start / resume rollout...")
        self.awaiting_start = True
        self.start_event.clear()
        self.start_event.wait()
        self.awaiting_start = False

    def close_client(self):
        """Stop and discard the cached RobotClient so the next run_inference creates a fresh one."""
        if self.cached_async_client is not None:
            try:
                self.cached_async_client.stop()
            except Exception as e:
                self.logger.warning(f"Error stopping cached client: {e}")
            self.cached_async_client = None
            self.cached_async_checkpoint_path = None
            self.logger.info("Cached client reset. Next run_inference will create a fresh connection.")

    def _joy_callback(self, msg):
        if len(msg.buttons) < 27:
            return
        if msg.buttons[2] and self.awaiting_start:
            self.start_event.set()
        if msg.buttons[8] and self.running and not self.awaiting_result:
            self.stop_event.set()
        if self.awaiting_result:
            if msg.buttons[12]:
                self.is_success = True
            elif msg.buttons[13]:
                self.is_success = False

    def run_inference(self):
        from std_msgs.msg import Empty

        if self.running:
            self.logger.warning("Rollout already in progress, ignoring command")
            return

        checkpoint_path = self.args.checkpoint_path
        task = self.args.task
        policy_type = self.args.policy_type

        if not checkpoint_path or not task:
            self.logger.error("Missing checkpoint_path or task")
            return

        self.logger.info(f"Rollout: policy_type={policy_type}, checkpoint={checkpoint_path}, task='{task}', max_steps={self.args.max_rollout_steps or 'unlimited'}")

        result_file = Path(f"logs/stagecraft/{self.args.experiment_name}/{policy_type}_result.log")
        result_file.parent.mkdir(parents=True, exist_ok=True)

        self.wait_for_start()

        self.running = True
        self.stop_event.clear()

        try:
            client = None
            if self.cached_async_checkpoint_path == checkpoint_path and self.cached_async_client is not None:
                self.logger.info(f"Reusing cached async client for {checkpoint_path}")
                client = self.cached_async_client
                client.reset()

            client = run_async_inference(
                self.robot_config, self.args,
                self.logger, self.stop_event, client=client,
            )

            self.cached_async_checkpoint_path = checkpoint_path
            self.cached_async_client = client
        except Exception as e:
            self.logger.error(f"Inference error: {e}")

        try:
            self.logger.info("Press spacemouse button 1 for success, button 2 for failure...")
            self.awaiting_result = True
            self.is_success = None
            while self.is_success is None:
                time.sleep(0.1)

            result = "success" if self.is_success else "failure"
            with open(result_file, "a") as f:
                f.write(checkpoint_path + "\n")
                f.write(task + "\n")
                f.write(result + "\n\n")
            self.logger.info(f"Logged result: {result}")
            self.send_reset()
        finally:
            self.awaiting_result = False
            self.running = False


if __name__ == "__main__":
    main()
