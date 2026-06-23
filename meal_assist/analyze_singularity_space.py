# -*- coding: utf-8 -*-
"""Numerically map singular and non-singular regions of the MuJoCo robot.

This script samples joint configurations q, evaluates the task Jacobian with
MuJoCo, and classifies each sample by SVD metrics:

    sigma_min(J(q)), condition(J(q)), manipulability(J(q)).

It is intentionally numerical, not symbolic. The default task Jacobian matches
the project convention:

    J_task = [ Jp_spoon_tip ; orientation_scale * Jr_link7 ]

where Jp_spoon_tip controls the spoon-tip linear velocity and Jr_link7 controls
the end-effector/spoon angular velocity.

Examples:
    python analyze_singularity_space.py --samples 20000
    python analyze_singularity_space.py --mode slice --joint-a 1 --joint-b 2 --grid 121
    python analyze_singularity_space.py --viewer --samples 5000 --color-by sigma
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from meal_assist.config import SystemConfig
from meal_assist.robot import RobotModel, mujoco


@dataclass
class SingularityConfig:
    mode: str = "random"
    samples: int = 20000
    grid: int = 121
    joint_a: int = 1
    joint_b: int = 2
    seed: int = 13
    margin_ratio: float = 0.015
    orientation_scale: float = 0.15
    sigma_threshold: float = 0.004
    condition_threshold: float = 900.0
    include_contacts: bool = False
    viewer: bool = False
    viewer_points: int = 5000
    viewer_point_size: float = 0.006
    color_by: str = "singular"
    out_dir: str = "singularity_analysis_output"


def task_jacobian(robot: RobotModel, d, orientation_scale: float) -> np.ndarray:
    """Return project task Jacobian [tip linear; scaled link7 angular]."""
    Jp, Jr = robot.jacobians(d)
    return np.vstack([Jp, orientation_scale * Jr])


def svd_metrics(J: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    """Return sigma_min, condition number, manipulability, all singular values."""
    S = np.linalg.svd(J, compute_uv=False)
    sigma_min = float(np.min(S))
    sigma_max = float(np.max(S))
    condition = float(sigma_max / (sigma_min + 1e-12))
    manipulability = float(np.prod(S))
    return sigma_min, condition, manipulability, S


def joint_names(model) -> List[str]:
    names: List[str] = []
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        names.append(name or f"joint_{j}")
    return names


def random_samples(robot: RobotModel, cfg: SingularityConfig) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    lo, hi = safe_bounds(robot, cfg.margin_ratio)
    for _ in range(cfg.samples):
        yield rng.uniform(lo, hi)


def slice_samples(robot: RobotModel, cfg: SingularityConfig) -> Iterable[np.ndarray]:
    lo, hi = safe_bounds(robot, cfg.margin_ratio)
    q_center = 0.5 * (lo + hi)
    qa_values = np.linspace(lo[cfg.joint_a], hi[cfg.joint_a], cfg.grid)
    qb_values = np.linspace(lo[cfg.joint_b], hi[cfg.joint_b], cfg.grid)
    for qa in qa_values:
        for qb in qb_values:
            q = q_center.copy()
            q[cfg.joint_a] = qa
            q[cfg.joint_b] = qb
            yield q


def safe_bounds(robot: RobotModel, margin_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    span = np.maximum(robot.q_upper - robot.q_lower, 1e-9)
    margin = margin_ratio * span
    return robot.q_lower + margin, robot.q_upper - margin


def evaluate_samples(robot: RobotModel, cfg: SingularityConfig) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    d = mujoco.MjData(robot.model)
    names = joint_names(robot.model)
    rows: List[Dict[str, float]] = []

    if cfg.mode == "slice":
        sample_iter = slice_samples(robot, cfg)
        total = cfg.grid * cfg.grid
    else:
        sample_iter = random_samples(robot, cfg)
        total = cfg.samples

    singular_count = 0
    contact_count = 0
    best_sigma = -np.inf
    worst_sigma = np.inf
    worst_condition = -np.inf
    best_row: Dict[str, float] | None = None
    worst_row: Dict[str, float] | None = None

    for idx, q in enumerate(sample_iter):
        robot.set_q(d, q)
        contact = int(d.ncon)
        if contact > 0:
            contact_count += 1
            if not cfg.include_contacts:
                continue

        J = task_jacobian(robot, d, cfg.orientation_scale)
        sigma_min, condition, manipulability, S = svd_metrics(J)
        is_singular = (
            sigma_min < cfg.sigma_threshold
            or condition > cfg.condition_threshold
        )
        if is_singular:
            singular_count += 1

        tip = robot.tip_pos(d)
        head_drop = robot.spoon_head_drop(d)
        margin = robot.min_joint_limit_margin_ratio(q)
        row: Dict[str, float] = {
            "sample": float(idx),
            "singular": float(is_singular),
            "sigma_min": sigma_min,
            "condition": condition,
            "manipulability": manipulability,
            "joint_margin_ratio": margin,
            "contact": float(contact),
            "tip_x": float(tip[0]),
            "tip_y": float(tip[1]),
            "tip_z": float(tip[2]),
            "head_drop": float(head_drop),
        }
        for i, val in enumerate(q):
            key = names[i] if i < len(names) else f"q{i + 1}"
            row[key] = float(val)
        for i, val in enumerate(S):
            row[f"sigma_{i + 1}"] = float(val)
        rows.append(row)

        if sigma_min > best_sigma:
            best_sigma = sigma_min
            best_row = row
        if sigma_min < worst_sigma:
            worst_sigma = sigma_min
            worst_row = row
        worst_condition = max(worst_condition, condition)

        if (idx + 1) % max(1, total // 10) == 0:
            print(f"[progress] {idx + 1}/{total} samples evaluated")

    valid = len(rows)
    summary = {
        "mode": cfg.mode,
        "requested_samples": float(total),
        "valid_samples": float(valid),
        "contact_skipped": float(contact_count if not cfg.include_contacts else 0),
        "contact_seen": float(contact_count),
        "singular_count": float(sum(int(r["singular"]) for r in rows)),
        "singular_ratio": float(sum(int(r["singular"]) for r in rows) / max(valid, 1)),
        "sigma_min_min": float(min((r["sigma_min"] for r in rows), default=np.nan)),
        "sigma_min_median": float(np.median([r["sigma_min"] for r in rows])) if rows else float("nan"),
        "sigma_min_max": float(max((r["sigma_min"] for r in rows), default=np.nan)),
        "condition_median": float(np.median([r["condition"] for r in rows])) if rows else float("nan"),
        "condition_max": float(worst_condition if rows else np.nan),
        "best_sigma_sample": float(best_row["sample"]) if best_row else float("nan"),
        "worst_sigma_sample": float(worst_row["sample"]) if worst_row else float("nan"),
    }
    return rows, summary


def write_outputs(rows: List[Dict[str, float]], summary: Dict[str, float], cfg: SingularityConfig) -> Path:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"singularity_{cfg.mode}.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        npz_path = out_dir / f"singularity_{cfg.mode}.npz"
        numeric = {k: np.array([row[k] for row in rows], dtype=float) for k in fieldnames}
        np.savez_compressed(npz_path, **numeric)

    with (out_dir / f"summary_{cfg.mode}.json").open("w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "summary": summary}, f, indent=2)

    try:
        write_plots(rows, cfg, out_dir)
    except Exception as exc:
        print(f"[WARN] plot generation failed: {exc}")

    return out_dir


def row_color(row: Dict[str, float], cfg: SingularityConfig) -> np.ndarray:
    """Return RGBA color for a sampled workspace point."""
    if cfg.color_by == "sigma":
        # Blue/cyan = better conditioned, red = near singular.
        ratio = np.clip(row["sigma_min"] / max(cfg.sigma_threshold * 5.0, 1e-9), 0.0, 1.0)
        return np.array([1.0 - ratio, 0.20 + 0.65 * ratio, 0.15 + 0.85 * ratio, 0.85], dtype=np.float32)
    if cfg.color_by == "condition":
        ratio = np.clip(row["condition"] / max(cfg.condition_threshold, 1e-9), 0.0, 1.0)
        return np.array([ratio, 0.80 * (1.0 - ratio), 1.0 - ratio, 0.85], dtype=np.float32)
    if row["singular"] >= 0.5:
        return np.array([1.0, 0.05, 0.03, 0.92], dtype=np.float32)
    return np.array([0.05, 0.75, 1.0, 0.55], dtype=np.float32)


def launch_viewer(robot: RobotModel, rows: List[Dict[str, float]], cfg: SingularityConfig) -> None:
    """Show sampled tip positions in MuJoCo viewer as workspace point cloud."""
    if not rows:
        print("[WARN] no rows to visualize")
        return
    if not hasattr(mujoco, "viewer"):
        raise RuntimeError("mujoco.viewer is not available in this environment")

    # Keep most informative points if there are too many for the debug scene:
    # prioritize singular/near-singular samples, then fill with regular samples.
    limit = max(1, int(cfg.viewer_points))
    ordered = sorted(rows, key=lambda r: (r["singular"], -r["condition"]), reverse=True)
    shown = ordered[: min(limit, len(ordered))]

    d = robot.data
    mat = np.eye(3, dtype=np.float64).reshape(-1)
    point_size = np.array([cfg.viewer_point_size, 0.0, 0.0], dtype=np.float64)

    singular_n = sum(1 for r in shown if r["singular"] >= 0.5)
    print(
        f"[VIEWER] showing {len(shown)} workspace points "
        f"({singular_n} singular/near-singular prioritized)"
    )
    print("[VIEWER] red = singular flag, cyan = non-singular unless --color-by changes it")
    print("[VIEWER] close the MuJoCo viewer window to finish")

    with mujoco.viewer.launch_passive(robot.model, d) as viewer:
        while viewer.is_running():
            scn = viewer.user_scn
            scn.ngeom = 0
            for row in shown:
                if scn.ngeom >= scn.maxgeom:
                    break
                pos = np.array([row["tip_x"], row["tip_y"], row["tip_z"]], dtype=np.float64)
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    point_size,
                    pos,
                    mat,
                    row_color(row, cfg),
                )
                scn.ngeom += 1
            viewer.sync()


def write_plots(rows: List[Dict[str, float]], cfg: SingularityConfig, out_dir: Path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    sigma = np.array([r["sigma_min"] for r in rows], dtype=float)
    condition = np.array([r["condition"] for r in rows], dtype=float)

    plt.figure(figsize=(8, 5))
    plt.hist(sigma, bins=80)
    plt.axvline(cfg.sigma_threshold, color="r", linestyle="--", label="sigma threshold")
    plt.xlabel("sigma_min(J)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"sigma_hist_{cfg.mode}.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    clipped = np.clip(condition, 0, np.percentile(condition, 99))
    plt.hist(clipped, bins=80)
    plt.axvline(cfg.condition_threshold, color="r", linestyle="--", label="condition threshold")
    plt.xlabel("condition number, clipped at p99")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"condition_hist_{cfg.mode}.png", dpi=160)
    plt.close()

    if cfg.mode == "slice":
        q_cols = [k for k in rows[0].keys() if k.startswith("joint") or k.startswith("q")]
        # Prefer numeric qN fallback if joint names are not simple.
        if cfg.joint_a < len(q_cols) and cfg.joint_b < len(q_cols):
            qa_key = q_cols[cfg.joint_a]
            qb_key = q_cols[cfg.joint_b]
        else:
            qa_key = f"q{cfg.joint_a + 1}"
            qb_key = f"q{cfg.joint_b + 1}"
        qa = np.array([r[qa_key] for r in rows], dtype=float)
        qb = np.array([r[qb_key] for r in rows], dtype=float)
        singular = np.array([r["singular"] for r in rows], dtype=float)

        plt.figure(figsize=(7, 6))
        sc = plt.scatter(qa, qb, c=sigma, s=8, cmap="viridis")
        plt.colorbar(sc, label="sigma_min(J)")
        plt.xlabel(qa_key)
        plt.ylabel(qb_key)
        plt.tight_layout()
        plt.savefig(out_dir / f"slice_sigma_j{cfg.joint_a}_j{cfg.joint_b}.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 6))
        sc = plt.scatter(qa, qb, c=singular, s=8, cmap="coolwarm", vmin=0, vmax=1)
        plt.colorbar(sc, label="singular flag")
        plt.xlabel(qa_key)
        plt.ylabel(qb_key)
        plt.tight_layout()
        plt.savefig(out_dir / f"slice_flag_j{cfg.joint_a}_j{cfg.joint_b}.png", dpi=180)
        plt.close()

        # True grid heatmaps for easier reading than scatter plots.
        qa_unique = np.unique(qa)
        qb_unique = np.unique(qb)
        if len(qa_unique) * len(qb_unique) == len(rows):
            sigma_grid = np.full((len(qa_unique), len(qb_unique)), np.nan)
            cond_grid = np.full_like(sigma_grid, np.nan)
            flag_grid = np.full_like(sigma_grid, np.nan)
            ia = {float(v): i for i, v in enumerate(qa_unique)}
            ib = {float(v): i for i, v in enumerate(qb_unique)}
            for row in rows:
                a = ia[float(row[qa_key])]
                b = ib[float(row[qb_key])]
                sigma_grid[a, b] = row["sigma_min"]
                cond_grid[a, b] = min(row["condition"], cfg.condition_threshold * 1.5)
                flag_grid[a, b] = row["singular"]

            extent = [qb_unique[0], qb_unique[-1], qa_unique[0], qa_unique[-1]]

            plt.figure(figsize=(7, 6))
            im = plt.imshow(
                sigma_grid,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="viridis",
            )
            plt.colorbar(im, label="sigma_min(J)")
            plt.xlabel(qb_key)
            plt.ylabel(qa_key)
            plt.title("Singularity heatmap: sigma_min")
            plt.tight_layout()
            plt.savefig(out_dir / f"heatmap_sigma_j{cfg.joint_a}_j{cfg.joint_b}.png", dpi=180)
            plt.close()

            plt.figure(figsize=(7, 6))
            im = plt.imshow(
                cond_grid,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="magma",
            )
            plt.colorbar(im, label=f"condition number, clipped at {cfg.condition_threshold * 1.5:g}")
            plt.xlabel(qb_key)
            plt.ylabel(qa_key)
            plt.title("Singularity heatmap: condition")
            plt.tight_layout()
            plt.savefig(out_dir / f"heatmap_condition_j{cfg.joint_a}_j{cfg.joint_b}.png", dpi=180)
            plt.close()

            plt.figure(figsize=(7, 6))
            im = plt.imshow(
                flag_grid,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="coolwarm",
                vmin=0,
                vmax=1,
            )
            plt.colorbar(im, label="singular flag")
            plt.xlabel(qb_key)
            plt.ylabel(qa_key)
            plt.title("Singularity heatmap: binary flag")
            plt.tight_layout()
            plt.savefig(out_dir / f"heatmap_flag_j{cfg.joint_a}_j{cfg.joint_b}.png", dpi=180)
            plt.close()


def parse_args() -> SingularityConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["random", "slice"], default="random")
    p.add_argument("--samples", type=int, default=20000)
    p.add_argument("--grid", type=int, default=121)
    p.add_argument("--joint-a", type=int, default=1, help="0-based joint index for slice mode")
    p.add_argument("--joint-b", type=int, default=2, help="0-based joint index for slice mode")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--margin-ratio", type=float, default=0.015)
    p.add_argument("--orientation-scale", type=float, default=0.15)
    p.add_argument("--sigma-threshold", type=float, default=0.004)
    p.add_argument("--condition-threshold", type=float, default=900.0)
    p.add_argument("--include-contacts", action="store_true")
    p.add_argument("--viewer", action="store_true", help="open MuJoCo viewer and render sampled tip positions")
    p.add_argument("--viewer-points", type=int, default=5000, help="maximum point-cloud samples shown in viewer")
    p.add_argument("--viewer-point-size", type=float, default=0.006)
    p.add_argument("--color-by", choices=["singular", "sigma", "condition"], default="singular")
    p.add_argument("--out-dir", default="singularity_analysis_output")
    return SingularityConfig(**vars(p.parse_args()))


def main() -> None:
    cfg = parse_args()
    system_cfg = SystemConfig()
    robot = RobotModel(system_cfg)

    if robot.model.nq != 6:
        print(f"[WARN] expected 6 DoF, but model.nq={robot.model.nq}")
    if cfg.mode == "slice":
        if not (0 <= cfg.joint_a < robot.model.nq and 0 <= cfg.joint_b < robot.model.nq):
            raise ValueError("joint-a and joint-b must be valid 0-based joint indices")
        if cfg.joint_a == cfg.joint_b:
            raise ValueError("joint-a and joint-b must be different")

    print("[INFO] numerical singularity analysis")
    print(f"[INFO] xml={system_cfg.xml_path}")
    print(f"[INFO] nq={robot.model.nq}, joint_names={joint_names(robot.model)}")
    print(f"[INFO] mode={cfg.mode}, orientation_scale={cfg.orientation_scale}")
    print(f"[INFO] singular if sigma_min < {cfg.sigma_threshold} or condition > {cfg.condition_threshold}")

    rows, summary = evaluate_samples(robot, cfg)
    out_dir = write_outputs(rows, summary, cfg)

    print("[SUMMARY]")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"[DONE] outputs written to {out_dir.resolve()}")
    if cfg.viewer:
        launch_viewer(robot, rows, cfg)


if __name__ == "__main__":
    main()
