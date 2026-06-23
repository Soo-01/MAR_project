# -*- coding: utf-8 -*-
"""JSON/CSV storage for scoop primitives and connector caches."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import SystemConfig
from .datatypes import FoodRegion, MouthConnector, NeutralConnector, ScoopPrimitive


class PrimitiveDatabase:
    """Persist and query generated scoop primitives."""

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.cfg.out_dir / "scoop_primitives.json"
        self.csv_path = self.cfg.out_dir / "scoop_primitives_summary.csv"

    def save(self, primitives: List[ScoopPrimitive], regions: List[FoodRegion]):
        """Save the full LUT as JSON and a readable summary as CSV."""
        payload = {
            "config": {
                "tray_x_length": self.cfg.tray_x_length,
                "tray_y_length": self.cfg.tray_y_length,
                "base_y_offset": self.cfg.base_y_offset,
                "tray_surface_z": self.cfg.tray_surface_z,
            },
            "regions": [asdict(r) for r in regions],
            "primitives": [asdict(p) for p in primitives],
        }
        self.json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with self.csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "primitive_id",
                    "region_id",
                    "food_x",
                    "food_y",
                    "drag_length",
                    "score",
                    "max_pos_error",
                    "max_tilt_error",
                    "preview_max_tilt",
                    "min_sigma",
                    "max_condition",
                    "max_contact",
                    "head_drop_pre",
                    "head_drop_engage",
                    "head_drop_drag_start",
                    "head_drop_drag_end",
                    "min_head_drop",
                    "max_head_drop_error",
                ]
            )
            for p in primitives:
                writer.writerow(
                    [
                        p.primitive_id,
                        p.region_id,
                        p.food_xy[0],
                        p.food_xy[1],
                        p.drag_length,
                        p.score,
                        p.max_pos_error,
                        p.max_tilt_error,
                        p.preview_max_tilt,
                        p.min_sigma,
                        p.max_condition,
                        p.max_contact,
                        p.head_drop_pre,
                        p.head_drop_engage,
                        p.head_drop_drag_start,
                        p.head_drop_drag_end,
                        p.min_head_drop,
                        p.max_head_drop_error,
                    ]
                )
        print("[SAVE]", self.json_path)
        print("[SAVE]", self.csv_path)

    def load(self) -> List[ScoopPrimitive]:
        """Load the full primitive LUT from JSON."""
        if not self.json_path.exists():
            raise FileNotFoundError(
                f"LUT JSON file not found. Run --mode build_lut first: {self.json_path}"
            )
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        return [ScoopPrimitive(**p) for p in payload["primitives"]]

    def load_summary_rows(self) -> List[Dict[str, str]]:
        """Load the CSV summary produced by ``build_lut``.

        The summary is used for ranking and filtering. The actual joint
        trajectories are still loaded from the JSON by ``primitive_id``.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"LUT CSV file not found. Run --mode build_lut first: {self.csv_path}"
            )
        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def select(
        self,
        region_id: int,
        food_xy: Tuple[float, float],
        top_k: int = 5,
    ) -> List[ScoopPrimitive]:
        """Return the closest low-score primitives for a requested food point."""
        primitives = [p for p in self.load() if p.region_id == region_id]
        if not primitives:
            return []
        f = np.array(food_xy, dtype=float)

        def key(p: ScoopPrimitive):
            dist = float(np.linalg.norm(np.array(p.food_xy) - f))
            return (dist, p.score)

        return sorted(primitives, key=key)[:top_k]

    def auto_select_from_lut(
        self,
        region_id: Optional[int] = None,
        n_actions: Optional[int] = 1,
        seed: Optional[int] = None,
        shuffle: bool = False,
        unique_food_xy: bool = True,
    ) -> List[ScoopPrimitive]:
        """Select executable primitives from the saved LUT summary.

        Selection ranks by score, optionally filters by region, optionally keeps
        only the best primitive per food sample, and then resolves each selected
        ``primitive_id`` back to the JSON trajectory.
        """
        rows = self.load_summary_rows()
        if region_id is not None:
            rows = [r for r in rows if int(r["region_id"]) == int(region_id)]
        if not rows:
            return []

        rows = sorted(
            rows,
            key=lambda r: (float(r["score"]), int(r["region_id"]), r["primitive_id"]),
        )

        if unique_food_xy:
            best: Dict[Tuple[int, float, float], Dict[str, str]] = {}
            for r in rows:
                key = (
                    int(r["region_id"]),
                    round(float(r["food_x"]), 6),
                    round(float(r["food_y"]), 6),
                )
                if key not in best or float(r["score"]) < float(best[key]["score"]):
                    best[key] = r
            rows = sorted(
                best.values(),
                key=lambda r: (
                    float(r["score"]),
                    int(r["region_id"]),
                    r["primitive_id"],
                ),
            )

        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(rows)

        if n_actions is None:
            n_actions = len(rows)
        selected_rows = [rows[i % len(rows)] for i in range(int(n_actions))]

        primitive_by_id = {p.primitive_id: p for p in self.load()}
        selected: List[ScoopPrimitive] = []
        missing = []
        for r in selected_rows:
            pid = r["primitive_id"]
            p = primitive_by_id.get(pid)
            if p is None:
                missing.append(pid)
            else:
                selected.append(p)
        if missing:
            print(f"[WARN] {len(missing)} CSV primitive IDs missing from JSON: {missing[:5]}")
        return selected


class MouthConnectorDatabase:
    """Persist the neutral-to-mouth connector trajectory."""

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.cfg.out_dir / "mouth_connector.json"

    def save(self, connector: MouthConnector):
        """Save the mouth connector with the config fields that affect staleness."""
        payload = {
            "config": {
                "mouth_y_range": self.cfg.mouth_y_range,
                "mouth_candidate_z_range": self.cfg.mouth_candidate_z_range,
                "mouth_forward_world": self.cfg.mouth_forward_world,
            },
            "connector": asdict(connector),
        }
        self.json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("[SAVE]", self.json_path)

    def load(self) -> MouthConnector:
        """Load the cached mouth connector."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Mouth connector LUT file not found: {self.json_path}")
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        return MouthConnector(**payload["connector"])

    def load_payload(self) -> Tuple[MouthConnector, Dict[str, object]]:
        """Load connector data together with its saved config snapshot."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Mouth connector LUT file not found: {self.json_path}")
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        return MouthConnector(**payload["connector"]), dict(payload.get("config", {}))


def _neutral_config_snapshot(cfg: SystemConfig) -> Dict[str, object]:
    """Return config fields that affect the neutral IK/cache result."""
    return {
        "neutral_radius": cfg.neutral_radius,
        "neutral_z_min": cfg.neutral_z_min,
        "neutral_z_max": cfg.neutral_z_max,
        "neutral_grid_xy": cfg.neutral_grid_xy,
        "neutral_grid_z": cfg.neutral_grid_z,
        "neutral_normal_weight": cfg.neutral_normal_weight,
        "neutral_head_drop_min": cfg.neutral_head_drop_min,
        "neutral_head_drop_weight": cfg.neutral_head_drop_weight,
        "neutral_position_tol": cfg.neutral_position_tol,
        "neutral_tilt_score_weight": cfg.neutral_tilt_score_weight,
        "neutral_joint_margin_ratio": cfg.neutral_joint_margin_ratio,
        "neutral_center_tray": cfg.neutral_center_tray,
        "neutral_margin_score_weight": cfg.neutral_margin_score_weight,
        "multi_start_trials": cfg.multi_start_trials,
        "random_seed": cfg.random_seed,
        "base_x_offset": cfg.base_x_offset,
        "base_y_offset": cfg.base_y_offset,
        "base_z_offset": cfg.base_z_offset,
        "tray_x_length": cfg.tray_x_length,
        "tray_y_length": cfg.tray_y_length,
        "tray_surface_z": cfg.tray_surface_z,
    }


class NeutralDatabase:
    """Persist the computed neutral configuration in ``neutral.json``."""

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.cfg.out_dir / "neutral.json"

    def save(self, connector: NeutralConnector):
        """Save neutral pose data with its config snapshot."""
        payload = {
            "config": _neutral_config_snapshot(self.cfg),
            "connector": asdict(connector),
        }
        self.json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("[SAVE]", self.json_path)

    def load_payload(self) -> Tuple[NeutralConnector, Dict[str, object]]:
        """Load neutral connector data and the saved config snapshot."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Neutral LUT file not found: {self.json_path}")
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        return NeutralConnector(**payload["connector"]), dict(payload.get("config", {}))
