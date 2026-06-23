# -*- coding: utf-8 -*-
"""Joint-space path planner used as a collision-aware connector.

The LUT stores key joint configurations such as scoop, lift, neutral, and
mouth poses. A straight joint interpolation between two keyframes can collide
with the tray or robot. This planner first checks the straight segment; if it
is blocked, it searches for intermediate waypoints with a simple RRT-style
tree and then shortcuts the result.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .config import SystemConfig
from .robot import RobotModel, mujoco


class JointPathPlanner:
    """Plan feasible joint-space waypoint paths between two configurations."""

    def __init__(self, cfg: SystemConfig, robot: RobotModel, seed: Optional[int] = None):
        self.cfg = cfg
        self.robot = robot
        self.rng = np.random.default_rng(cfg.random_seed if seed is None else seed)
        self._d = mujoco.MjData(robot.model)
        self.lo = robot.q_lower.copy()
        self.hi = robot.q_upper.copy()
        self.span = np.maximum(self.hi - self.lo, 1e-6)
        self.nq = robot.model.nq

    def state_free(self, q: np.ndarray) -> bool:
        """Return True when ``q`` is inside joint limits and collision-free."""
        if not self.robot.is_joint_limit_safe(q):
            return False
        self.robot.set_q(self._d, q)
        return int(self._d.ncon) <= self.cfg.contact_allowed

    def segment_free(self, q0: np.ndarray, q1: np.ndarray, max_step: float = 0.04) -> bool:
        """Check the full straight joint segment by discretized collision tests."""
        diff = q1 - q0
        steps = max(2, int(np.ceil(float(np.max(np.abs(diff))) / max_step)))
        for i in range(steps + 1):
            if not self.state_free(q0 + diff * (i / steps)):
                return False
        return True

    def _sample(self) -> np.ndarray:
        """Sample a random joint vector away from the hard joint limits."""
        m = self.cfg.joint_limit_margin_ratio
        return self.rng.uniform(self.lo + m * self.span, self.hi - m * self.span)

    def _nearest(self, tree: List[np.ndarray], q: np.ndarray) -> int:
        """Return the index of the nearest tree node in Euclidean joint space."""
        return int(np.argmin([float(np.dot(n - q, n - q)) for n in tree]))

    def _steer(self, q_from: np.ndarray, q_to: np.ndarray, step: float) -> np.ndarray:
        """Move from ``q_from`` toward ``q_to`` by at most ``step`` per joint."""
        diff = q_to - q_from
        d = float(np.max(np.abs(diff)))
        return q_to.copy() if d <= step else q_from + diff * (step / d)

    def plan(
        self,
        q0: np.ndarray,
        q1: np.ndarray,
        max_iter: int = 4000,
        step: float = 0.25,
        goal_bias: float = 0.2,
        max_step_check: float = 0.04,
    ) -> Optional[List[np.ndarray]]:
        """Plan a collision-free waypoint list from ``q0`` to ``q1``.

        Returns ``[q0, q1]`` when the straight segment is already feasible,
        otherwise returns a shortcut RRT path. ``None`` means no path was found
        within ``max_iter`` iterations.
        """
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        if not self.state_free(q0) or not self.state_free(q1):
            return None
        if self.segment_free(q0, q1, max_step_check):
            return [q0.copy(), q1.copy()]

        tree: List[np.ndarray] = [q0.copy()]
        parents: List[int] = [-1]
        for _ in range(max_iter):
            q_rand = q1 if self.rng.random() < goal_bias else self._sample()
            idx = self._nearest(tree, q_rand)
            q_new = self._steer(tree[idx], q_rand, step)
            if not self.segment_free(tree[idx], q_new, max_step_check):
                continue
            tree.append(q_new)
            parents.append(idx)
            if self.segment_free(q_new, q1, max_step_check):
                tree.append(q1.copy())
                parents.append(len(tree) - 2)
                return self._shortcut(
                    self._trace(tree, parents, len(tree) - 1),
                    max_step_check,
                )
        return None

    def _trace(self, tree, parents, idx) -> List[np.ndarray]:
        """Recover a path by following parent indices back to the root."""
        path = []
        i = idx
        while i != -1:
            path.append(tree[i])
            i = parents[i]
        return path[::-1]

    def _shortcut(
        self,
        path: List[np.ndarray],
        max_step_check: float,
        rounds: int = 300,
    ) -> List[np.ndarray]:
        """Remove unnecessary intermediate waypoints when direct jumps are safe."""
        path = [np.asarray(p, dtype=float) for p in path]
        for _ in range(rounds):
            n = len(path)
            if n <= 2:
                break
            i = int(self.rng.integers(0, n - 1))
            j = int(self.rng.integers(i + 2, n)) if n - i > 2 else -1
            if j < 0:
                continue
            if self.segment_free(path[i], path[j], max_step_check):
                path = path[: i + 1] + path[j:]
        return path

    @staticmethod
    def path_length(path: List[np.ndarray]) -> float:
        """Return the Euclidean length of a joint-space waypoint path."""
        return float(
            sum(np.linalg.norm(path[k + 1] - path[k]) for k in range(len(path) - 1))
        )
