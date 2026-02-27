#!/usr/bin/env python3
"""
Click-to-pick: left-click a pixel in the live camera image to pick at that
3-D point, then place at the hardcoded pose below.

Usage:
    python test_pick_place.py

Controls:
    Left-click  — pick at the clicked 3-D point, then place
    q           — quit
"""

import math
import time
import threading
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from primitive_actions.action import Pick, Place
from pointcloud_camera import RealSensePointCloudCamera
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig


@dataclass
class PrimitiveParam:
    x:               float
    y:               float
    z:               float
    yaw:             float
    approach_height: float


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class PickPlaceHandler(Node):

    def __init__(self, camera: RealSensePointCloudCamera):
        super().__init__('pick_place_handler')
        self._pick_client  = ActionClient(self, Pick,  'pick')
        self._place_client = ActionClient(self, Place, 'place')

        # ------------------------------------------------------------------
        # Configuration — edit these values
        # ------------------------------------------------------------------

        # Throw place pose
        self.throw_place_pose = PrimitiveParam(
            x=-0.1, y=-0.6, z=0.30,
            yaw=-math.pi / 2,
            approach_height=0.55,
        )

        # Pick defaults (x/y/z come from the mouse click)
        self.pick_defaults = PrimitiveParam(
            x=0.0, y=0.0, z=0.10,   # overwritten by click
            yaw=0.0,
            approach_height=0.30,
        )

        # Camera
        self._camera = camera

        # Private executor used only to wait on action futures.
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Wait for both action servers, then connect the camera."""
        self.get_logger().info('Waiting for /pick server …')
        self._pick_client.wait_for_server()
        self.get_logger().info('Waiting for /place server …')
        self._place_client.wait_for_server()
        self._camera.connect()
        self.get_logger().info('Connected.')

    def disconnect(self) -> None:
        """Disconnect the camera."""
        self._camera.disconnect()
        self.get_logger().info('Camera disconnected.')

    # ------------------------------------------------------------------
    # Primitives — fully parameterised on x, y, z, yaw, approach_height
    # ------------------------------------------------------------------

    def send_pick(self, pose: PrimitiveParam) -> bool:
        goal = Pick.Goal()
        goal.x               = float(pose.x)
        goal.y               = float(pose.y)
        goal.z               = float(pose.z)
        goal.yaw             = float(pose.yaw)
        goal.approach_height = float(pose.approach_height)

        self.get_logger().info(
            f'Pick  x={pose.x:.4f}  y={pose.y:.4f}  z={pose.z:.4f}  '
            f'yaw={pose.yaw:.4f}  approach_h={pose.approach_height:.4f}')

        future = self._pick_client.send_goal_async(
            goal, feedback_callback=self._pick_feedback_cb)
        rclpy.spin_until_future_complete(self, future, executor=self._executor)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Pick goal rejected!')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, executor=self._executor)
        result = result_future.result().result

        if result.success:
            self.get_logger().info(f'Pick SUCCESS: {result.message}')
        else:
            self.get_logger().error(f'Pick FAILED:  {result.message}')
        return result.success

    def send_place(self, pose: PrimitiveParam) -> bool:
        goal = Place.Goal()
        goal.x               = float(pose.x)
        goal.y               = float(pose.y)
        goal.z               = float(pose.z)
        goal.yaw             = float(pose.yaw)
        goal.approach_height = float(pose.approach_height)

        self.get_logger().info(
            f'Place x={pose.x:.4f}  y={pose.y:.4f}  z={pose.z:.4f}  '
            f'yaw={pose.yaw:.4f}  approach_h={pose.approach_height:.4f}')

        future = self._place_client.send_goal_async(
            goal, feedback_callback=self._place_feedback_cb)
        rclpy.spin_until_future_complete(self, future, executor=self._executor)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Place goal rejected!')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, executor=self._executor)
        result = result_future.result().result

        if result.success:
            self.get_logger().info(f'Place SUCCESS: {result.message}')
        else:
            self.get_logger().error(f'Place FAILED:  {result.message}')
        return result.success

    # ------------------------------------------------------------------
    # Feedback callbacks
    # ------------------------------------------------------------------

    def _pick_feedback_cb(self, feedback_msg):
        self.get_logger().info(f'[Pick ] stage: {feedback_msg.feedback.stage}')

    def _place_feedback_cb(self, feedback_msg):
        self.get_logger().info(f'[Place] stage: {feedback_msg.feedback.stage}')


# ---------------------------------------------------------------------------
# Camera + click UI
# ---------------------------------------------------------------------------

def run_click_ui(node: PickPlaceHandler) -> None:
    """
    Show a live colour image.
    Left-click a pixel → pick at that 3-D robot-frame point → place at the
    hardcoded pose.
    Press 'q' to quit.
    """
    latest: dict = {"data": None}
    busy = threading.Event()

    WIN = "Click to pick  |  'q' to quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if busy.is_set():
            print("Robot is busy — click ignored.")
            return

        data = latest["data"]
        if data is None:
            return

        # Remap click from rotated display back to original camera frame
        h, w = data.rgb.shape[:2]
        orig_x = w - 1 - x
        orig_y = h - 1 - y

        pt = data.point_at(orig_x, orig_y)
        if np.isnan(pt).any():
            print(f"pixel ({x}, {y}) → no valid depth, skipping.")
            return

        px, py, pz = float(pt[0]), float(pt[1]), float(pt[2])
        print(
            f"pixel ({x:4d}, {y:4d}) → "
            f"X={px:+.4f}  Y={py:+.4f}  Z={pz:+.4f}  (m, panda_link0)"
        )

        def run_sequence():
            busy.set()
            try:
                pick_pose = PrimitiveParam(
                    x=px+0.03, y=py+0.025, z=max(pz-0.02, 0.05),
                    yaw=node.pick_defaults.yaw,
                    approach_height=node.pick_defaults.approach_height,
                )
                ok = node.send_pick(pick_pose)
                if not ok:
                    print("Pick failed — aborting place.")
                    return

                node.send_place(node.throw_place_pose)
            finally:
                busy.clear()

        threading.Thread(target=run_sequence, daemon=True).start()

    cv2.setMouseCallback(WIN, on_click)

    try:
        while True:
            t0 = time.perf_counter()

            data = node._camera.get_data()
            latest["data"] = data

            bgr = cv2.rotate(cv2.cvtColor(data.rgb, cv2.COLOR_RGB2BGR), cv2.ROTATE_180)

            # Status overlay
            if busy.is_set():
                label, colour = "BUSY", (0, 0, 255)
            else:
                label, colour = "Click to pick", (0, 255, 0)
            cv2.putText(bgr, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)

            cv2.imshow(WIN, bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            elapsed = time.perf_counter() - t0
            if elapsed < 1 / 30:
                time.sleep(1 / 30 - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)

    t_base_cam = RealSensePointCloudCamera.make_transform(
        1.41104,   0.155109,  1.0711,               # translation  (metres)
       -0.625165,  0.650804,  0.253996, -0.348007,  # quaternion   x y z w
    )

    config = RealSenseCameraConfig(
        serial_number_or_name="938422074102",
        width=640,
        height=480,
        fps=30,
        use_depth=True,
    )

    camera = RealSensePointCloudCamera(RealSenseCamera(config), T_base_cam=t_base_cam)
    node = PickPlaceHandler(camera)
    node.connect()
    print("Ready.  Left-click on the object you want to pick up.")
    try:
        run_click_ui(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.disconnect()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()