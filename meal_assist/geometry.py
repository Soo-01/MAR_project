# -*- coding: utf-8 -*-
"""Tray geometry utilities.

This module converts between the tray frame and the MuJoCo world frame, and
defines the default food regions used by the scoop primitive builder.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .config import SystemConfig
from .datatypes import FoodRegion


class TrayGeometry:
    """Map tray-frame coordinates to robot/world-frame coordinates.

    Current assumption:
    - The MuJoCo world frame is the robot base frame.
    - The tray origin is translated from the robot base by
      ``[base_x_offset, base_y_offset, base_z_offset]``.
    - Tray axes are parallel to world axes.

    If the real tray has yaw/roll/pitch relative to the robot base, replace
    ``R_world_tray`` with the measured rotation matrix.
    """

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.p_world_tray_origin = np.array(
            [cfg.base_x_offset, cfg.base_y_offset, cfg.base_z_offset],
            dtype=float,
        )
        self.R_world_tray = np.eye(3)

    def tray_to_world(self, p_tray: np.ndarray) -> np.ndarray:
        """Convert a point from tray coordinates to world coordinates."""
        return self.p_world_tray_origin + self.R_world_tray @ p_tray

    def world_to_tray(self, p_world: np.ndarray) -> np.ndarray:
        """Convert a point from world coordinates to tray coordinates."""
        return self.R_world_tray.T @ (p_world - self.p_world_tray_origin)

    def default_regions(self) -> List[FoodRegion]:
        """Return a provisional five-region tray layout.

        The polygons are normalized by ``tray_x_length`` and ``tray_y_length``.
        Replace these boundaries with measured tray/food-compartment polygons
        when real hardware dimensions are available.
        """
        Lx = self.cfg.tray_x_length
        Ly = self.cfg.tray_y_length
        regions = [
            FoodRegion(
                1,
                "rice_or_main",
                [
                    (0.00 * Lx, 0.00 * Ly),
                    (0.52 * Lx, 0.00 * Ly),
                    (0.52 * Lx, 0.48 * Ly),
                    (0.00 * Lx, 0.48 * Ly),
                ],
                barrier_x=0.00 * Lx,
                barrier_y_min=0.00 * Ly,
                barrier_y_max=0.48 * Ly,
                barrier_height=0.025,
                barrier_thickness=0.006,
            ),
            FoodRegion(
                2,
                "side_1",
                [
                    (0.52 * Lx, 0.00 * Ly),
                    (1.00 * Lx, 0.00 * Ly),
                    (1.00 * Lx, 0.32 * Ly),
                    (0.52 * Lx, 0.32 * Ly),
                ],
                barrier_x=0.52 * Lx,
                barrier_y_min=0.00 * Ly,
                barrier_y_max=0.32 * Ly,
                barrier_height=0.025,
                barrier_thickness=0.006,
            ),
            FoodRegion(
                3,
                "side_2",
                [
                    (0.52 * Lx, 0.32 * Ly),
                    (1.00 * Lx, 0.32 * Ly),
                    (1.00 * Lx, 0.64 * Ly),
                    (0.52 * Lx, 0.64 * Ly),
                ],
                barrier_x=0.52 * Lx,
                barrier_y_min=0.32 * Ly,
                barrier_y_max=0.64 * Ly,
                barrier_height=0.025,
                barrier_thickness=0.006,
            ),
            FoodRegion(
                4,
                "side_3",
                [
                    (0.52 * Lx, 0.64 * Ly),
                    (1.00 * Lx, 0.64 * Ly),
                    (1.00 * Lx, 1.00 * Ly),
                    (0.52 * Lx, 1.00 * Ly),
                ],
                barrier_x=0.52 * Lx,
                barrier_y_min=0.64 * Ly,
                barrier_y_max=1.00 * Ly,
                barrier_height=0.025,
                barrier_thickness=0.006,
            ),
            FoodRegion(
                5,
                "soup_or_extra",
                [
                    (0.00 * Lx, 0.48 * Ly),
                    (0.52 * Lx, 0.48 * Ly),
                    (0.52 * Lx, 1.00 * Ly),
                    (0.00 * Lx, 1.00 * Ly),
                ],
                barrier_x=0.00 * Lx,
                barrier_y_min=0.48 * Ly,
                barrier_y_max=1.00 * Ly,
                barrier_height=0.025,
                barrier_thickness=0.006,
            ),
        ]
        return regions

    def neutral_points_world(self) -> List[np.ndarray]:
        """Sample candidate neutral positions in world coordinates.

        The neutral region is a small cylinder in tray coordinates centered at
        ``neutral_center_tray``. IK chooses the candidate with low position
        error, small tilt, and comfortable joint-limit margin.
        """
        cx, cy = self.cfg.neutral_center_tray
        center_tray = np.array([cx, cy, 0.0])
        pts = []
        xy = np.linspace(
            -self.cfg.neutral_radius,
            self.cfg.neutral_radius,
            self.cfg.neutral_grid_xy,
        )
        zs = np.linspace(self.cfg.neutral_z_min, self.cfg.neutral_z_max, self.cfg.neutral_grid_z)
        for dx in xy:
            for dy in xy:
                if dx * dx + dy * dy <= self.cfg.neutral_radius**2:
                    for z in zs:
                        p_tray = center_tray + np.array(
                            [dx, dy, self.cfg.tray_surface_z + z]
                        )
                        pts.append(self.tray_to_world(p_tray))
        return pts

    def neutral_pos_world(self) -> np.ndarray:
        """Return the legacy single neutral point in world coordinates."""
        p_tray = np.array(self.cfg.neutral_pos_tray, dtype=float)
        return self.tray_to_world(p_tray)
