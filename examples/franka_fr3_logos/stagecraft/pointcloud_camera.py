#!/usr/bin/env python

# ---------------------------------------------------------------------------
# Copyright (c) 2025 Prabin Kumar Rath
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ---------------------------------------------------------------------------

"""
Franka FR3 — live ordered pointcloud viewer using open3d.

Usage:
    python extract_pointcloud.py
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation

from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig


@dataclass
class PointCloudData:
    """
    Holds a synchronised colour image and its corresponding ordered pointcloud
    in the robot base frame (panda_link0).

    Attributes
    ----------
    rgb : (H, W, 3) uint8
        Colour image in RGB format.
    xyz : (H, W, 3) float32
        3-D point for each pixel in panda_link0 (metres).
        Pixels with no valid depth are NaN.
        Query the 3-D point for image pixel (x, y) with ``xyz[y, x]``
        or ``point_at(x, y)``.
    """

    rgb: np.ndarray   # (H, W, 3) uint8
    xyz: np.ndarray   # (H, W, 3) float32  — panda_link0

    def point_at(self, x: int, y: int) -> np.ndarray:
        """Return the 3-D point (X, Y, Z) in metres in panda_link0 for image pixel (x, y)."""
        return self.xyz[y, x]


class RealSensePointCloudCamera:
    """
    Wraps a RealSenseCamera and provides a ``get_data()`` method that returns
    a colour image and its pixel-aligned ordered pointcloud in one call.

    Every pixel ``(x, y)`` in the returned image maps to the 3-D point
    ``data.xyz[y, x]`` (or equivalently ``data.point_at(x, y)``).

    Parameters
    ----------
    config : RealSenseCameraConfig
        Camera configuration.  ``use_depth`` is forced to ``True``
        automatically; all other fields (serial number, resolution, fps,
        rotation …) are respected as-is.

    Example
    -------
    >>> cam = RealSensePointCloudCamera(config)
    >>> cam.connect()
    >>> data = cam.get_data()
    >>> xyz_at_centre = data.point_at(x=320, y=240)
    >>> cam.disconnect()

    Can also be used as a context manager::

        with RealSensePointCloudCamera(config) as cam:
            data = cam.get_data()
    """

    def __init__(self, config: RealSenseCameraConfig,
                 T_base_cam: np.ndarray) -> None:
        """
        Parameters
        ----------
        config : RealSenseCameraConfig
        T_base_cam : (4, 4) float64 homogeneous transform
            Transforms points from the colour optical frame into the robot
            base frame (panda_link0).  Use ``T_BASE_CAM`` for the
            eye-to-hand calibration stored in this module.
        """
        object.__setattr__(config, "use_depth", True)
        self._camera = RealSenseCamera(config)
        self._aligner: rs.align | None = None
        self._T_base_cam = np.asarray(T_base_cam, dtype=np.float64)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_transform(tx: float, ty: float, tz: float,
                       qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """Build a 4x4 homogeneous transform from a translation and unit quaternion (x,y,z,w)."""
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = [tx, ty, tz]
        return T

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._camera.connect()
        self._aligner = rs.align(rs.stream.color)

    def disconnect(self) -> None:
        self._camera.disconnect()
        self._aligner = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def get_data(self) -> PointCloudData:
        """
        Capture one frame and return the colour image together with its
        pixel-aligned ordered pointcloud.

        Returns
        -------
        PointCloudData
            ``.rgb``  — (H, W, 3) uint8, RGB colour image
            ``.xyz``  — (H, W, 3) float32, 3-D point per pixel in the
                        colour optical frame (metres); NaN where depth is
                        invalid or out of [0.1 m, 5.0 m].
        """
        if self._aligner is None:
            raise RuntimeError("Camera is not connected. Call connect() first.")

        camera = self._camera

        ret, frameset = camera.rs_pipeline.try_wait_for_frames(timeout_ms=500)
        if not ret:
            raise RuntimeError("Timed out waiting for frameset.")

        # Align depth into the colour sensor's frame so every (u, v) in both
        # images corresponds to exactly the same physical ray.
        aligned = self._aligner.process(frameset)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        color = np.asanyarray(color_frame.get_data())   # (H, W, 3) RGB uint8
        depth = np.asanyarray(depth_frame.get_data())   # (H, W)    uint16 mm

        # Apply any rotation / colour-mode conversion configured on the camera.
        color = camera._postprocess_image(color)
        depth = camera._postprocess_image(depth, depth_frame=True)

        # Colour-stream intrinsics: after alignment depth lives in the colour
        # sensor's coordinate frame, so these are the correct intrinsics for
        # backprojection.
        intr = (
            camera.rs_profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

        H, W = depth.shape
        depth_m = depth.astype(np.float32) * 0.001     # mm → metres

        # Pixel grid — u is the column index (x), v is the row index (y).
        u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                           np.arange(H, dtype=np.float32))

        X = (u - intr.ppx) * depth_m / intr.fx
        Y = (v - intr.ppy) * depth_m / intr.fy
        Z = depth_m

        # xyz[row, col] == xyz[y, x] → 3-D point for image pixel (x, y)
        xyz_cam = np.stack([X, Y, Z], axis=-1)          # (H, W, 3) float32
        invalid = (depth_m < 0.1) | (depth_m > 5.0)
        xyz_cam[invalid] = np.nan

        # Transform every point into the robot base frame (panda_link0).
        pts = xyz_cam.reshape(-1, 3).astype(np.float64)
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        pts_base = (self._T_base_cam @ np.hstack([pts, ones]).T).T[:, :3]
        xyz = pts_base.reshape(H, W, 3).astype(np.float32)
        xyz[invalid] = np.nan

        return PointCloudData(rgb=color, xyz=xyz)


def visualize_open3d(cam: RealSensePointCloudCamera) -> None:
    """Open an Open3D window streaming the live coloured pointcloud."""
    vis = o3d.visualization.Visualizer()
    vis.create_window("Franka FR3 - Ordered Pointcloud", width=1280, height=720)
    pcd = o3d.geometry.PointCloud()
    added = False

    try:
        while True:
            t0 = time.perf_counter()

            data = cam.get_data()

            valid = ~np.isnan(data.xyz[:, :, 0])
            pcd.points = o3d.utility.Vector3dVector(data.xyz[valid].astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(
                data.rgb[valid].astype(np.float64) / 255.0
            )

            if not added:
                vis.add_geometry(pcd)
                vis.get_view_control().set_zoom(0.4)
                added = True
            else:
                vis.update_geometry(pcd)

            if not vis.poll_events():
                break
            vis.update_renderer()

            elapsed = time.perf_counter() - t0
            if elapsed < 1 / 30:
                time.sleep(1 / 30 - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()


def visualize_cv2_click(cam: RealSensePointCloudCamera) -> None:
    """Show a live colour image; click any pixel to print its 3-D point."""
    latest: dict = {"data": None}

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        data = latest["data"]
        if data is None:
            return
        pt = data.point_at(x, y)
        if np.isnan(pt).any():
            print(f"pixel ({x}, {y}) → no valid depth")
            return
        print(
            f"pixel ({x:4d}, {y:4d}) → "
            f"X={pt[0]:+.4f}  Y={pt[1]:+.4f}  Z={pt[2]:+.4f}  (m, panda_link0)"
        )

    cv2.namedWindow("Franka FR3 - Pointcloud", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Franka FR3 - Pointcloud", on_click)

    try:
        while True:
            t0 = time.perf_counter()

            data = cam.get_data()
            latest["data"] = data

            bgr = cv2.cvtColor(data.rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("Franka FR3 - Pointcloud", bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            elapsed = time.perf_counter() - t0
            if elapsed < 1 / 30:
                time.sleep(1 / 30 - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


def main():
    # -------------------------------------------------------------------------
    # Eye-to-hand calibration result (ROS static_transform_publisher convention)
    #   parent: panda_link0  →  child: camera_color_optical_frame
    #   args: x y z  qx qy qz qw
    # -------------------------------------------------------------------------
    T_base_cam = RealSensePointCloudCamera.make_transform(
        1.41104,  0.155109, 1.0711,              # translation  (metres)
        -0.625165, 0.650804, 0.253996, -0.348007,  # quaternion  x y z w
    )

    config = RealSenseCameraConfig(
        serial_number_or_name="938422074102",
        width=640,
        height=480,
        fps=30,
        use_depth=True,
    )

    with RealSensePointCloudCamera(config, T_base_cam=T_base_cam) as cam:
        print("Camera connected.")

        # ── choose one viewer ──────────────────────────────────────────
        # visualize_open3d(cam)        # 3-D pointcloud viewer (Open3D)
        visualize_cv2_click(cam)       # click-to-query 3-D point (cv2)
        # ──────────────────────────────────────────────────────────────

    print("Done.")


if __name__ == "__main__":
    main()
