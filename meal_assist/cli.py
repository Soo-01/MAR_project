# -*- coding: utf-8 -*-
"""Command-line entry points for building, selecting, and replaying LUTs."""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import time

from .config import SystemConfig
from .database import MouthConnectorDatabase, PrimitiveDatabase
from .datatypes import FoodRegion, ScoopPrimitive, StepResult
from .geometry import TrayGeometry
from .ik import IKSolver
from .neutral import compute_q_neutral, load_or_build_q_neutral
from .robot import RobotModel, mujoco
from .runner import SequenceRunner
from .scoop_builder import ScoopPrimitiveBuilder

def _build_and_save_mouth_connector(cfg: SystemConfig, robot: RobotModel, ik: IKSolver) -> bool:
    """q_neutral을 seed로 mouth connector를 계산해 mouth_connector.json에 저장한다.

    build_lut와 build_mouth_lut가 공유한다. q_neutral이 없으면 건너뛴다.
    """
    if robot.q_neutral is None:
        print("[MOUTH CONNECTOR SKIP] q_neutral이 없어 connector를 빌드하지 않습니다.")
        return False
    runner_for_mouth = SequenceRunner(cfg, robot)
    ok_connector, connector = runner_for_mouth.build_mouth_connector(
        seed_q=robot.q_neutral,
        ik=ik,
        rng=np.random.default_rng(cfg.random_seed),
    )
    if ok_connector and connector is not None:
        MouthConnectorDatabase(cfg).save(connector)
        return True
    print("[MOUTH CONNECTOR FAIL] connector 빌드 실패 — 저장하지 않습니다.")
    return False


def build_mouth_lut(cfg: SystemConfig):
    """scoop LUT 재빌드 없이 mouth connector LUT만 생성/갱신한다."""
    tray = TrayGeometry(cfg)
    robot = RobotModel(cfg)
    ik = IKSolver(cfg, robot)
    load_or_build_q_neutral(cfg, tray, robot, ik, force_rebuild=True)
    ok = _build_and_save_mouth_connector(cfg, robot, ik)
    print("[BUILD MOUTH LUT]", "완료" if ok else "실패")


def build_lut(cfg: SystemConfig, regions_filter: Optional[List[int]] = None):
    tray = TrayGeometry(cfg)
    robot = RobotModel(cfg)
    ik = IKSolver(cfg, robot)
    # Build and cache the neutral pose used by replay and mouth transfer.
    load_or_build_q_neutral(cfg, tray, robot, ik, force_rebuild=True)
    _build_and_save_mouth_connector(cfg, robot, ik)
    builder = ScoopPrimitiveBuilder(cfg, tray, robot, ik)
    regions = tray.default_regions()
    if regions_filter is not None:
        wanted = {int(r) for r in regions_filter}
        regions = [r for r in regions if r.region_id in wanted]

    all_primitives: List[ScoopPrimitive] = []
    for r in regions:
        print("==========================================")
        print(f"[BUILD REGION] {r.region_id}: {r.name}")
        print("polygon:", r.polygon_xy)
        ps = builder.build_for_region(r)
        print(f"[REGION DONE] region={r.region_id}, primitives={len(ps)}")
        all_primitives.extend(ps)

    db = PrimitiveDatabase(cfg)
    db.save(all_primitives, regions)
    print("[TOTAL PRIMITIVES]", len(all_primitives))


def select_primitive(cfg: SystemConfig, region: int, food_xy: Tuple[float, float], top_k: int = 5) -> List[ScoopPrimitive]:
    db = PrimitiveDatabase(cfg)
    selected = db.select(region, food_xy, top_k=top_k)
    if not selected:
        print(f"[NO CANDIDATE] region={region}, food_xy={food_xy}")
        return []
    print("==========================================")
    print("[SELECTED PRIMITIVES]")
    for i, p in enumerate(selected, start=1):
        dist = np.linalg.norm(np.array(p.food_xy) - np.array(food_xy))
        print(f"rank={i} id={p.primitive_id} region={p.region_id} dist={dist:.4f} score={p.score:.4f} food_xy={p.food_xy} drag={p.drag_length}")
    return selected


def replay_selected(
    cfg: SystemConfig,
    region: int,
    food_xy: Tuple[float, float],
    viewer: bool,
    seed: Optional[int] = None,
    start_from_random: bool = True,
):
    """지정된 food_xy에 대한 top-1 primitive를 전체 sequence로 실행.

    제어 코드 계획서 + 사용자 요구사항:
        Random Initial -> Neutral -> Pre -> Engage -> Drag(-X) -> Lift
        -> Neutral -> Mouth -> Neutral
    """
    selected = select_primitive(cfg, region, food_xy, top_k=1)
    if not selected:
        return
    robot = RobotModel(cfg)
    # Load or build the neutral pose before replay.
    tray = TrayGeometry(cfg)
    ik_for_neutral = IKSolver(cfg, robot)
    load_or_build_q_neutral(cfg, tray, robot, ik_for_neutral)
    runner = SequenceRunner(cfg, robot)
    runner.trail_positions.clear()

    primitive = selected[0]
    print("\n[RUN SCOOP SEQUENCE]")
    print("0. Random Initial Position")
    print("1. Random -> Neutral")
    print("2. Neutral -> Pre-scoop")
    print("3. Pre-scoop -> Engage")
    print("4. Engage -> Drag start")
    print("5. Drag start -> Drag end (-X scoop)")
    print("6. Drag end -> Lift")
    print("7. Lift -> Neutral")
    print("8. Neutral -> Mouth")
    print("9. Mouth -> Neutral")

    if viewer:
        with mujoco.viewer.launch_passive(robot.model, robot.data) as v:
            completed = _execute_primitives_continuously(
                cfg, robot, runner, [primitive], v=v,
                start_from_random=start_from_random, seed=seed,
                use_full_sequence=True,
            )
    else:
        completed = _execute_primitives_continuously(
            cfg, robot, runner, [primitive], v=None,
            start_from_random=start_from_random, seed=seed,
            use_full_sequence=True,
        )
    print("[DONE] replay 완료" if completed else "[ABORTED] replay 중단")


def _best_primitives_for_region(cfg: SystemConfig, region: int) -> List[ScoopPrimitive]:
    """LUT에서 region별 food_xy마다 score가 가장 낮은 primitive만 추린다."""
    db = PrimitiveDatabase(cfg)
    all_primitives = [p for p in db.load() if p.region_id == region]
    best_per_food: Dict[Tuple[float, float], ScoopPrimitive] = {}
    for p in all_primitives:
        key = tuple(p.food_xy)
        if key not in best_per_food or p.score < best_per_food[key].score:
            best_per_food[key] = p
    return list(best_per_food.values())


def _execute_primitives_continuously(
    cfg: SystemConfig,
    robot: RobotModel,
    runner: SequenceRunner,
    selected: List[ScoopPrimitive],
    v=None,
    neutral_between: bool = True,           # legacy 옵션 (use_full_sequence=True면 자동 포함)
    start_from_random: bool = True,         # 사용자 요구: Random 초기 위치에서 시작
    seed: Optional[int] = None,
    use_full_sequence: bool = True,         # 사용자 요구: Mouth 단계 포함한 전체 sequence
    random_pose_hold_frames: int = 60,      # 랜덤 초기 자세를 viewer에 잠깐 보여주는 시간
):
    """여러 primitive를 하나의 MuJoCo data / viewer에서 연속 실행한다.

    제어 코드 계획서 (260515 계획서 7번) 기준 전체 sequence:
        Random Initial Point
            -> Neutral
            -> [ Pre-scoop -> Engage -> Drag start -> Drag end (-X) -> Lift
                 -> Neutral -> Mouth -> Neutral ] x N
    """
    # ---- 1. Random Initial Position에서 시작 ----
    completed_all = False

    if start_from_random:
        rng = np.random.default_rng(seed)
        q_random = runner.sample_neutral_reachable_initial_q(rng)
        robot.set_q(robot.data, q_random)
        if robot.model.nu > 0:
            robot.data.ctrl[:robot.model.nu] = q_random[:robot.model.nu]
        print(f"\n[STEP 0] Random Initial Position 설정 (q[:4]={np.round(q_random[:4], 3).tolist()})")
        if v is not None:
            # 잠깐 랜덤 초기 자세를 보여주고 trail에 시작점 기록
            try:
                runner.trail_positions.append(robot.tip_pos(robot.data).copy())
            except Exception:
                pass
            for _ in range(max(1, random_pose_hold_frames)):
                runner._render_trail(v)
                v.sync()
                time.sleep(robot.model.opt.timestep)
        # 2. Random -> Neutral
        print("[STEP 1] Random Initial -> Neutral")
        neutral_result = runner.replay_neutral(v=v, frames=max(960, 4 * cfg.frames_per_segment))
        if not neutral_result.ok:
            print("[ABORT] 초기 Neutral 도달 실패. scoop sequence를 시작하지 않습니다.")
            return False
    else:
        robot.set_q(robot.data, robot.q_center.copy())
        if robot.model.nu > 0:
            robot.data.ctrl[:robot.model.nu] = robot.q_center[:robot.model.nu]

    # ---- 3. 각 primitive: Neutral -> Pre -> Engage -> Drag -> Lift -> Neutral -> Mouth -> Neutral ----
    ik = IKSolver(cfg, robot) if use_full_sequence else None
    for i, primitive in enumerate(selected, 1):
        print(
            f"\n[ACTION {i}/{len(selected)}] {primitive.primitive_id}  "
            f"region={primitive.region_id}  food={primitive.food_xy}  score={primitive.score:.4f}"
        )
        if use_full_sequence:
            # 한 primitive 안에서 Mouth/Neutral 복귀까지 포함됨
            result = runner.replay_full_sequence(primitive, v=v, ik=ik, neutral_after_lift=True)
            if not result.ok:
                print(f"[ABORT] action {i} 실패. 다음 scoop으로 진행하지 않습니다.")
                return False
        else:
            result = runner.replay_continuous(primitive, v=v)
            if not result.ok:
                print(f"[ABORT] action {i} 실패. 다음 scoop으로 진행하지 않습니다.")
                return False
            if neutral_between and i < len(selected):
                print("[NEUTRAL >>] 다음 동작 전 neutral 복귀...")
                neutral_result = runner.replay_neutral(v=v, frames=max(960, 4 * cfg.frames_per_segment))
                if not neutral_result.ok:
                    print("[ABORT] 다음 동작 전 Neutral 도달 실패. 다음 scoop으로 진행하지 않습니다.")
                    return False

    completed_all = True
    return completed_all


def run_lut_sequence(
    cfg: SystemConfig,
    region: Optional[int] = None,
    n_actions: Optional[int] = 1,
    viewer: bool = False,
    seed: Optional[int] = None,
    shuffle: bool = False,
):
    """build_lut 결과 CSV에서 실행할 primitive를 자동으로 가져와 연속 실행한다.

    사용자가 food_xy를 직접 넣지 않아도 된다. CSV summary는 실행 후보의
    primitive_id와 score를 고르는 index로 사용하고, 실제 q_pre~q_lift는
    같은 LUT의 JSON에서 가져온다.
    """
    db = PrimitiveDatabase(cfg)
    selected = db.auto_select_from_lut(
        region_id=region,
        n_actions=n_actions,
        seed=seed,
        shuffle=shuffle,
        unique_food_xy=True,
    )
    if not selected:
        target = "전체 LUT" if region is None else f"region {region}"
        print(f"[ERROR] {target}에서 실행할 primitive가 없습니다. 먼저 --mode build_lut를 실행하세요.")
        return

    print(f"\n{'='*70}")
    print("[RUN LUT SEQUENCE] CSV summary 기반 자동 선택")
    print(f"  csv : {db.csv_path}")
    print(f"  json: {db.json_path}")
    print(f"  region filter: {region if region is not None else 'ALL'}")
    print(f"  actions: {len(selected)}")
    print(f"{'='*70}")
    for i, p in enumerate(selected, 1):
        print(f"  action {i:2d}: id={p.primitive_id} region={p.region_id} food_xy={p.food_xy} drag={p.drag_length:.3f} score={p.score:.4f}")
    print(f"{'='*70}\n")

    robot = RobotModel(cfg)
    # Load or build the neutral pose before replay.
    tray_for_neutral = TrayGeometry(cfg)
    ik_for_neutral = IKSolver(cfg, robot)
    load_or_build_q_neutral(cfg, tray_for_neutral, robot, ik_for_neutral)
    runner = SequenceRunner(cfg, robot)
    runner.trail_positions.clear()

    if viewer:
        with mujoco.viewer.launch_passive(robot.model, robot.data) as v:
            completed = _execute_primitives_continuously(
                cfg, robot, runner, selected, v=v,
                start_from_random=True, seed=seed,
                use_full_sequence=True,
            )
    else:
        completed = _execute_primitives_continuously(
            cfg, robot, runner, selected, v=None,
            start_from_random=True, seed=seed,
            use_full_sequence=True,
        )

    print("\n" + "="*70)
    print("[DONE] CSV LUT 기반 sequence 실행 완료" if completed else "[ABORTED] CSV LUT 기반 sequence 중단")
    print("="*70)


def run_region(
    cfg: SystemConfig,
    region: int,
    n_actions: Optional[int] = None,
    viewer: bool = False,
    seed: Optional[int] = None,
):
    """지정 region 안의 여러 scoop primitive를 한 시뮬레이션에서 연속 실행한다."""
    pool = _best_primitives_for_region(cfg, region)
    if not pool:
        print(f"[ERROR] region {region}에 저장된 primitive가 없습니다. 먼저 --mode build_lut를 실행하세요.")
        return

    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    if n_actions is None:
        n_actions = len(pool)
    selected = [pool[i % len(pool)] for i in range(n_actions)]

    print(f"\n{'='*62}")
    print(f"[RUN REGION {region}] 저장 food 위치 수={len(pool)}, 수행 동작={n_actions}")
    print(f"{'='*62}")
    for i, p in enumerate(selected, 1):
        print(f"  action {i:2d}: {p.primitive_id} food_xy={p.food_xy} drag={p.drag_length:.3f} m score={p.score:.4f}")
    print(f"{'='*62}\n")

    robot = RobotModel(cfg)
    # Load or build the neutral pose before replay.
    tray_for_neutral = TrayGeometry(cfg)
    ik_for_neutral = IKSolver(cfg, robot)
    load_or_build_q_neutral(cfg, tray_for_neutral, robot, ik_for_neutral)
    runner = SequenceRunner(cfg, robot)
    runner.trail_positions.clear()

    if viewer:
        with mujoco.viewer.launch_passive(robot.model, robot.data) as v:
            completed = _execute_primitives_continuously(
                cfg, robot, runner, selected, v=v,
                start_from_random=True, seed=seed,
                use_full_sequence=True,
            )
    else:
        completed = _execute_primitives_continuously(
            cfg, robot, runner, selected, v=None,
            start_from_random=True, seed=seed,
            use_full_sequence=True,
        )

    print("\n" + "="*62)
    print(f"[DONE] region {region} {n_actions}회 scoop 완료" if completed else f"[ABORTED] region {region} sequence 중단")
    print("="*62)


def run_all_regions(
    cfg: SystemConfig,
    regions: List[int],
    n_actions_per_region: int = 1,
    viewer: bool = False,
    seed: Optional[int] = None,
):
    """Region 1~5의 대표 primitive를 한 viewer/data에서 순서대로 연속 실행한다."""
    rng = np.random.default_rng(seed)
    selected: List[ScoopPrimitive] = []

    for region in regions:
        pool = _best_primitives_for_region(cfg, region)
        if not pool:
            print(f"[WARN] region {region} primitive 없음. 건너뜀.")
            continue
        pool = sorted(pool, key=lambda p: p.score)
        if n_actions_per_region <= 1:
            chosen = [pool[0]]
        else:
            rng.shuffle(pool)
            pool = sorted(pool, key=lambda p: p.score)
            chosen = [pool[i % len(pool)] for i in range(n_actions_per_region)]
        selected.extend(chosen)

    if not selected:
        print("[ERROR] 실행할 primitive가 없습니다. 먼저 --mode build_lut를 실행하세요.")
        return

    print(f"\n{'='*70}")
    print(f"[RUN ALL REGIONS] regions={regions}, 총 동작 수={len(selected)}")
    print(f"{'='*70}")
    for i, p in enumerate(selected, 1):
        print(f"  action {i:2d}: region={p.region_id} id={p.primitive_id} food_xy={p.food_xy} drag={p.drag_length:.3f} score={p.score:.4f}")
    print(f"{'='*70}\n")

    robot = RobotModel(cfg)
    # Load or build the neutral pose before replay.
    tray_for_neutral = TrayGeometry(cfg)
    ik_for_neutral = IKSolver(cfg, robot)
    load_or_build_q_neutral(cfg, tray_for_neutral, robot, ik_for_neutral)
    runner = SequenceRunner(cfg, robot)
    runner.trail_positions.clear()

    if viewer:
        with mujoco.viewer.launch_passive(robot.model, robot.data) as v:
            completed = _execute_primitives_continuously(
                cfg, robot, runner, selected, v=v,
                start_from_random=True, seed=seed,
                use_full_sequence=True,
            )
    else:
        completed = _execute_primitives_continuously(
            cfg, robot, runner, selected, v=None,
            start_from_random=True, seed=seed,
            use_full_sequence=True,
        )

    print("\n" + "="*70)
    print(f"[DONE] regions {regions} 연속 scoop 완료" if completed else f"[ABORTED] regions {regions} sequence 중단")
    print("="*70)


def test_run(cfg: SystemConfig):
    """Run a lightweight validation without opening the viewer.

    Checks neutral-pose construction, mouth connector generation, and one
    representative primitive/boundary validation.
    """
    tray = TrayGeometry(cfg)
    robot = RobotModel(cfg)
    ik = IKSolver(cfg, robot)

    print("=" * 70)
    print("[TEST_RUN] mouth connector LUT + FK head_drop validation (no viewer)")
    print("=" * 70)

    # 1) q_neutral 계산
    q_neutral = compute_q_neutral(cfg, tray, robot, ik)
    if q_neutral is not None:
        d = mujoco.MjData(robot.model)
        robot.set_q(d, q_neutral)
        tip = robot.tip_pos(d)
        n_target = np.array(cfg.world_up, dtype=float)
        f_target = np.array(cfg.scoop_drag_direction_world, dtype=float)
        tilt, _fwd, _du, _df = robot.orientation_errors(d, n_target, f_target)
        neutral_err = float(np.linalg.norm(tip - tray.neutral_pos_world()))
        neutral_result = StepResult(
            label="TEST_Q_NEUTRAL",
            ok=neutral_err <= cfg.neutral_position_tol and robot.is_joint_limit_safe(q_neutral),
            reason="validated" if neutral_err <= cfg.neutral_position_tol else "neutral_position_error",
            target_pos=tuple(tray.neutral_pos_world().tolist()),
            actual_pos=tuple(tip.tolist()),
            pos_error=neutral_err,
            tilt_error=tilt,
            contact=int(d.ncon),
        )
        print(f"[q_neutral] joints={np.round(q_neutral, 4).tolist()}")
        print(f"[q_neutral] tip_world={tip.tolist()}, target_world={tray.neutral_pos_world().tolist()}")
        print(f"[q_neutral] joint_limit_safe (1.5%)={robot.is_joint_limit_safe(q_neutral)}")
        print(neutral_result.summary())
    else:
        print("[q_neutral] 계산 실패")
        print(StepResult(label="TEST_Q_NEUTRAL", ok=False, reason="neutral_ik_failed").summary())

    # 2) Mouth Z 후보 multi-start IK
    runner = SequenceRunner(cfg, robot)
    seed = q_neutral if q_neutral is not None else robot.q_center.copy()
    ok, q_mouth, metrics, pos_best = runner.solve_mouth_q_multi(seed_q=seed, ik=ik, rng=np.random.default_rng(0))
    if ok and q_mouth is not None:
        d = mujoco.MjData(robot.model)
        robot.set_q(d, q_mouth)
        tip = robot.tip_pos(d)
        mouth_err = float(np.linalg.norm(tip - np.array(pos_best, dtype=float)))
        mouth_result = StepResult(
            label="TEST_MOUTH_IK",
            ok=mouth_err <= cfg.mouth_position_tol and bool(metrics.get("joint_limit_ok", 0.0)),
            reason="validated" if mouth_err <= cfg.mouth_position_tol else "mouth_position_error",
            target_pos=tuple(pos_best),
            actual_pos=tuple(tip.tolist()),
            pos_error=mouth_err,
            tilt_error=float(metrics.get("tilt_error", 0.0)),
            contact=int(metrics.get("contact", 0)),
        )
        print(f"[MOUTH IK OK] chosen_pos={pos_best}, tip_world={tip.tolist()}")
        print(f"[MOUTH IK OK] metrics={ {k: float(v) for k, v in metrics.items()} }")
        print(mouth_result.summary())
        for i in range(robot.model.nq):
            lo, hi = robot.q_lower[i], robot.q_upper[i]
            d_to_lo = q_mouth[i] - lo
            d_to_hi = hi - q_mouth[i]
            ratio_lo = d_to_lo / max(hi - lo, 1e-6)
            ratio_hi = d_to_hi / max(hi - lo, 1e-6)
            tag = ""
            if ratio_lo <= cfg.joint_limit_margin_ratio:
                tag = "  <- near LO (strict 1.5% 위반)"
            elif ratio_hi <= cfg.joint_limit_margin_ratio:
                tag = "  <- near HI (strict 1.5% 위반)"
            print(
                f"  joint {i}: q={q_mouth[i]:+.4f}, "
                f"range=[{lo:+.4f}, {hi:+.4f}], "
                f"margin_lo_ratio={ratio_lo:.3%}, margin_hi_ratio={ratio_hi:.3%}{tag}"
            )
    else:
        print(f"[MOUTH IK FAIL] best metrics={metrics}, pos_best={pos_best}")
        print(StepResult(label="TEST_MOUTH_IK", ok=False, reason="mouth_ik_failed").summary())

    # 3) Scoop boundary 통과 여부 (LUT 첫 region에서 첫 primitive)
    try:
        db = PrimitiveDatabase(cfg)
        all_primitives = db.load()
    except FileNotFoundError:
        print("[TEST_RUN] LUT 미생성 — boundary check 생략 (build_lut 후 재실행)")
        return
    if not all_primitives:
        print("[TEST_RUN] LUT empty — boundary check skipped")
        return

    region_ids = sorted({p.region_id for p in all_primitives})
    print(f"[BOUNDARY CHECK] regions={region_ids}")
    for rid in region_ids:
        prims = [p for p in all_primitives if p.region_id == rid]
        if not prims:
            continue
        p0 = min(prims, key=lambda p: p.score)
        ds_world = np.array(p0.drag_start_pos)
        de_world = np.array(p0.drag_end_pos)
        ds_tray = tray.world_to_tray(ds_world)
        de_tray = tray.world_to_tray(de_world)
        in_tray = (
            cfg.spoon_outer_margin <= ds_tray[0] <= cfg.tray_x_length - cfg.spoon_outer_margin and
            cfg.spoon_outer_margin <= ds_tray[1] <= cfg.tray_y_length - cfg.spoon_outer_margin and
            cfg.spoon_outer_margin <= de_tray[0] <= cfg.tray_x_length - cfg.spoon_outer_margin and
            cfg.spoon_outer_margin <= de_tray[1] <= cfg.tray_y_length - cfg.spoon_outer_margin
        )
        print(
            f"  region {rid} primitive={p0.primitive_id}: "
            f"drag_start_tray={np.round(ds_tray, 4).tolist()}, "
            f"drag_end_tray={np.round(de_tray, 4).tolist()}, "
            f"within_tray={in_tray}"
        )

    print("=" * 70)
    print("[TEST_RUN] 완료")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="식사보조로봇 scoop primitive LUT 생성 / 선택 / 재생",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["build_lut", "build_mouth_lut", "select", "replay", "run_sequence", "run_lut", "run_region", "run_all_regions", "test_run"],
        required=True,
        help=(
            "build_lut    : scoop + mouth connector LUT 저장. 예: --regions 5 로 단일 region 먼저 확인\n"
            "build_mouth_lut : scoop LUT 재빌드 없이 mouth connector LUT만 생성/갱신\n"
            "select     : food_xy 지정 -> 후보 primitive 출력\n"
            "replay       : food_xy 지정 -> top-1 primitive를 StepResult 검증과 함께 전체 sequence 재생\n"
            "run_sequence : food_xy가 없으면 CSV LUT에서 자동 선택, 있으면 지정 food_xy 실행\n"
            "run_lut      : CSV summary에서 primitive를 자동 선택해 실행\n"
            "run_region   : 한 region 안에서 여러 primitive를 연속 실행. 먼저 확인 예: --region 5 --n_actions 1 --viewer\n"
            "run_all_regions : region 1~5 대표 primitive 연속 실행\n"
            "test_run     : Neutral/Mouth connector IK와 StepResult 출력 검증"
        ),
    )
    parser.add_argument("--region",    type=int,   default=None,
                        help="대상 region_id (1~5). run_sequence/run_lut에서 생략하면 전체 CSV 중 score 최저 후보 자동 선택")
    parser.add_argument("--food_xy",   type=float, nargs=2, default=None,
                        help="[select/replay 선택사항] tray frame 음식 위치 x y. 생략하면 run_sequence는 CSV LUT에서 자동 선택")
    parser.add_argument("--n_actions", type=int,   default=None,
                        help="[run_region] 수행할 scoop 횟수 / [run_all_regions] region당 동작 수")
    parser.add_argument("--regions", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="[build_lut/run_all_regions] 대상 region 목록. 기본: 1 2 3 4 5")
    parser.add_argument("--food_samples", type=int, default=None,
                        help="[build_lut 전용] region 내부 food sample 축 개수 override. 단일 region 확인 예: 3, 빠른 디버그 예: 1")
    parser.add_argument("--ik_iters", type=int, default=None,
                        help="[build_lut/test 전용] IK 최대 반복 수 override. 빠른 디버그 예: 250")
    parser.add_argument("--multi_start", type=int, default=None,
                        help="[build_lut/test 전용] multi-start random seed 수 override. 단일 region 확인 예: 12, 빠른 디버그 예: 1")
    parser.add_argument("--seed",      type=int,   default=None,
                        help="랜덤 시드 (재현용, 미지정 시 비결정적)")
    parser.add_argument("--shuffle",   action="store_true",
                        help="[run_lut/run_sequence 자동 선택] CSV 후보를 score 정렬 대신 shuffle해서 실행")
    parser.add_argument("--viewer",    action="store_true",
                        help="MuJoCo 뷰어 창 표시")
    parser.add_argument("--xml",       type=str,   default="robot_model_v5_scene.xml",
                        help="MuJoCo XML 모델 파일명")
    parser.add_argument("--out_dir",   type=str,   default=None,
                        help="LUT/output directory override")
    return parser.parse_args()


def main():
    # Windows 콘솔(cp949)에서 '—' 같은 비-cp949 문자 print 시 UnicodeEncodeError로
    # 죽는 것을 방지한다. 인코딩(cp949)은 유지하되 못 쓰는 문자만 '?'로 대체.
    import sys as _sys
    for _stream in (_sys.stdout, _sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass
    args = parse_args()
    cfg = SystemConfig(xml_name=args.xml)
    if args.out_dir is not None:
        cfg.out_dir_name = str(args.out_dir)
    if args.food_samples is not None:
        cfg.food_samples_per_region_axis = int(args.food_samples)
    if args.ik_iters is not None:
        cfg.ik_iters = int(args.ik_iters)
    if args.multi_start is not None:
        cfg.multi_start_trials = int(args.multi_start)
        cfg.mouth_multi_start_seeds = int(args.multi_start)
    if args.mode == "build_lut":
        build_lut(cfg, regions_filter=args.regions)
    elif args.mode == "build_mouth_lut":
        build_mouth_lut(cfg)
    elif args.mode == "select":
        if args.food_xy is None:
            print("[ERROR] select 모드는 --food_xy x y가 필요합니다.")
            return
        if args.region is None:
            print("[ERROR] select 모드는 --region이 필요합니다.")
            return
        select_primitive(cfg, args.region, tuple(args.food_xy), top_k=5)
    elif args.mode == "replay":
        if args.food_xy is None:
            run_lut_sequence(cfg, region=args.region, n_actions=(args.n_actions if args.n_actions is not None else 1), viewer=args.viewer, seed=args.seed, shuffle=args.shuffle)
        else:
            if args.region is None:
                print("[ERROR] replay 모드에서 --food_xy를 직접 주는 경우 --region도 필요합니다.")
                return
            replay_selected(cfg, args.region, tuple(args.food_xy), viewer=args.viewer, seed=args.seed, start_from_random=True)
    elif args.mode == "run_sequence":
        if args.food_xy is None:
            run_lut_sequence(cfg, region=args.region, n_actions=(args.n_actions if args.n_actions is not None else 1), viewer=args.viewer, seed=args.seed, shuffle=args.shuffle)
        else:
            replay_selected(cfg, args.region, tuple(args.food_xy), viewer=args.viewer, seed=args.seed, start_from_random=True)
    elif args.mode == "run_lut":
        run_lut_sequence(cfg, region=args.region, n_actions=(args.n_actions if args.n_actions is not None else 1), viewer=args.viewer, seed=args.seed, shuffle=args.shuffle)
    elif args.mode == "run_region":
        if args.region is None:
            print("[ERROR] run_region 모드는 --region이 필요합니다.")
            return
        run_region(
            cfg,
            region=args.region,
            n_actions=args.n_actions,
            viewer=args.viewer,
            seed=args.seed,
        )
    elif args.mode == "run_all_regions":
        run_all_regions(
            cfg,
            regions=args.regions,
            n_actions_per_region=(args.n_actions if args.n_actions is not None else 1),
            viewer=args.viewer,
            seed=args.seed,
        )
    elif args.mode == "test_run":
        test_run(cfg)
