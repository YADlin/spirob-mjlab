#!/usr/bin/env python3
"""Evaluate any checkpoint in the shared SpiRob polar-sector curriculum.

The same program evaluates Arc-1, Arc-2, and Full.  Every full run contains:

* 50 nominal repeats;
* 10 continuous samples in each of the 38 retained S2-C cells;
* 25 area-uniform samples in each of 25 radial/angular sector strata.

NaN rollouts remain visible and count as failures.  They do not automatically
invalidate the complete evaluation: the report uses a rate-based numerical
review trigger (0.1% by default) and separates protocol validity from policy
performance.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import platform
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.random import seed_rng
from mjlab.utils.torch import configure_torch_backends

import evaluate_stage1 as stage2_base
import mjlab.tasks  # noqa: F401  # Populate MJLab's built-in registry.
import spirob_mjlab  # noqa: F401  # Register the SpiRob tasks.
from spirob_mjlab.sector_curriculum import (
    NOMINAL_EGG_XY_M,
    RETENTION_CELL_LOWER_LEFT_MM,
    RETENTION_CELL_SIZE_MM,
    SECTOR_SPECS,
    STAGE2_SECTOR_ARC1_TASK_ID,
    SectorSpec,
    polar_to_xy_m,
    spawn_is_clear_xy_m,
)


PROTOCOL_VERSION = "S2-sector-evaluation-v1"
EVALUATION_SEED = 3210
RETENTION_POSITION_SEED = 3201
SECTOR_POSITION_SEED = 3202
RADIAL_STRATA = 5
ANGULAR_STRATA = 5

FULL_NOMINAL_REPEATS = 50
FULL_RETENTION_SAMPLES_PER_CELL = 10
FULL_SECTOR_SAMPLES_PER_STRATUM = 25
SMOKE_NOMINAL_REPEATS = 5
SMOKE_RETENTION_SAMPLES_PER_CELL = 1
SMOKE_SECTOR_SAMPLES_PER_STRATUM = 1

NOMINAL_SUCCESS_RATE_GUIDE = 0.84
RETENTION_SUCCESS_RATE_GUIDE = 0.90
SECTOR_SUCCESS_RATE_GUIDE = 0.80
NAN_RATE_REVIEW_TRIGGER = 0.001  # 0.1%; descriptive policy, not a physics law.

Mode = Literal["smoke", "full"]


@dataclass(frozen=True)
class SectorEvaluateConfig:
    """Inputs for one frozen-checkpoint sector evaluation."""

    run_id: str
    training_seed: int
    training_git_commit: str
    checkpoint_file: str
    expected_checkpoint_sha256: str
    output_dir: str
    task_id: str = STAGE2_SECTOR_ARC1_TASK_ID
    mode: Mode = "smoke"
    evaluation_seed: int = EVALUATION_SEED
    policy_mode: Literal["deterministic"] = "deterministic"
    device: str | None = None
    num_envs: int = 256
    allow_dirty_evaluator: bool = False


def _samples_for_mode(mode: Mode) -> tuple[int, int, int]:
    if mode == "full":
        return (
            FULL_NOMINAL_REPEATS,
            FULL_RETENTION_SAMPLES_PER_CELL,
            FULL_SECTOR_SAMPLES_PER_STRATUM,
        )
    return (
        SMOKE_NOMINAL_REPEATS,
        SMOKE_RETENTION_SAMPLES_PER_CELL,
        SMOKE_SECTOR_SAMPLES_PER_STRATUM,
    )


def _retention_conditions(samples_per_cell: int) -> list[stage2_base.Stage2Condition]:
    rng = np.random.default_rng(RETENTION_POSITION_SEED)
    conditions: list[stage2_base.Stage2Condition] = []
    for cell_index, (x_low_mm, y_low_mm) in enumerate(
        RETENTION_CELL_LOWER_LEFT_MM
    ):
        if samples_per_cell == 1:
            draws = np.full((1, 2), RETENTION_CELL_SIZE_MM / 2.0)
        else:
            draws = rng.uniform(
                0.0,
                RETENTION_CELL_SIZE_MM,
                size=(samples_per_cell, 2),
            )
        for repeat_index, (x_draw_mm, y_draw_mm) in enumerate(draws):
            conditions.append(
                stage2_base.Stage2Condition(
                    phase="retention_s2c_cells",
                    phase_episode_index=-1,
                    offset_x_mm=float(x_low_mm + x_draw_mm),
                    offset_y_mm=float(y_low_mm + y_draw_mm),
                    spatial_stratum=cell_index,
                    stratum_x_index=-1,
                    stratum_y_index=-1,
                    repeat_index=repeat_index,
                    is_unique_continuous_spawn=True,
                )
            )
    return conditions


def _sector_conditions(
    spec: SectorSpec,
    samples_per_stratum: int,
) -> list[stage2_base.Stage2Condition]:
    rng = np.random.default_rng(SECTOR_POSITION_SEED)
    radius_min, radius_max = spec.radius_range_m
    angle_min, angle_max = spec.angle_range_deg
    radial_width = (radius_max - radius_min) / RADIAL_STRATA
    angular_width = (angle_max - angle_min) / ANGULAR_STRATA
    conditions: list[stage2_base.Stage2Condition] = []

    for angular_index in range(ANGULAR_STRATA):
        theta_low = angle_min + angular_index * angular_width
        theta_high = theta_low + angular_width
        for radial_index in range(RADIAL_STRATA):
            r_low = radius_min + radial_index * radial_width
            r_high = r_low + radial_width
            stratum = angular_index * RADIAL_STRATA + radial_index
            accepted = 0
            attempts = 0
            while accepted < samples_per_stratum:
                attempts += 1
                if attempts > 100_000:
                    raise RuntimeError(
                        "Could not generate a collision-clear condition in "
                        f"sector stratum {stratum}"
                    )
                radius_m = math.sqrt(
                    r_low**2 + rng.random() * (r_high**2 - r_low**2)
                )
                angle_deg = rng.uniform(theta_low, theta_high)
                x_m, y_m = polar_to_xy_m(radius_m, angle_deg)
                if not spawn_is_clear_xy_m(x_m, y_m):
                    continue
                conditions.append(
                    stage2_base.Stage2Condition(
                        phase="active_polar_sector",
                        phase_episode_index=-1,
                        offset_x_mm=(x_m - NOMINAL_EGG_XY_M[0]) * 1000.0,
                        offset_y_mm=(y_m - NOMINAL_EGG_XY_M[1]) * 1000.0,
                        spatial_stratum=stratum,
                        stratum_x_index=radial_index,
                        stratum_y_index=angular_index,
                        repeat_index=accepted,
                        is_unique_continuous_spawn=True,
                    )
                )
                accepted += 1
    return conditions


def _evaluation_conditions(
    spec: SectorSpec,
    mode: Mode,
) -> list[stage2_base.Stage2Condition]:
    nominal_repeats, retention_samples, sector_samples = _samples_for_mode(mode)
    nominal = [
        stage2_base.Stage2Condition(
            phase="nominal_repeat",
            phase_episode_index=index,
            offset_x_mm=0.0,
            offset_y_mm=0.0,
            spatial_stratum=-1,
            stratum_x_index=-1,
            stratum_y_index=-1,
            repeat_index=index,
            is_unique_continuous_spawn=False,
        )
        for index in range(nominal_repeats)
    ]
    conditions = [
        *nominal,
        *_retention_conditions(retention_samples),
        *_sector_conditions(spec, sector_samples),
    ]
    rng = np.random.default_rng(EVALUATION_SEED)
    rng.shuffle(conditions)
    return [
        replace(condition, phase_episode_index=index)
        for index, condition in enumerate(conditions)
    ]


def _phase_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    return {
        name: sum(stage2_base._truth(row[name]) for row in rows)
        for name in ("success", "fall", "timeout", "oob", "nan")
    }


def _make_polar_heatmap(
    matrix: np.ndarray,
    spec: SectorSpec,
    output_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    radius_min, radius_max = spec.radius_range_m
    angle_min, angle_max = spec.angle_range_deg
    radial_edges_mm = np.linspace(radius_min * 1000.0, radius_max * 1000.0, 6)
    angular_edges = np.linspace(angle_min, angle_max, 6)
    radial_centres = 0.5 * (radial_edges_mm[:-1] + radial_edges_mm[1:])
    angular_centres = 0.5 * (angular_edges[:-1] + angular_edges[1:])

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    image = ax.imshow(
        matrix,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
        extent=(
            radial_edges_mm[0],
            radial_edges_mm[-1],
            angular_edges[0],
            angular_edges[-1],
        ),
        interpolation="nearest",
        aspect="auto",
    )
    for angular_index, theta in enumerate(angular_centres):
        for radial_index, radius in enumerate(radial_centres):
            ax.text(
                radius,
                theta,
                f"{100.0 * matrix[angular_index, radial_index]:.0f}%",
                ha="center",
                va="center",
                fontsize=9,
            )
    ax.set_title(f"{spec.label} success by polar stratum")
    ax.set_xlabel("Robot-centred radius (mm)")
    ax.set_ylabel("Angle from world +x (degrees)")
    ax.set_xticks(radial_centres)
    ax.set_yticks(angular_centres)
    fig.colorbar(image, ax=ax, label="Observed success fraction")
    fig.text(
        0.5,
        0.015,
        "Collision-invalid initial positions are excluded before evaluation.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    paths = [
        output_dir / "active_sector_success.png",
        output_dir / "active_sector_success.svg",
    ]
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def _summarize_retention(
    rows: list[dict[str, object]],
    samples_per_cell: int,
    output_dir: Path,
) -> tuple[dict[str, object], list[Path]]:
    selected = [row for row in rows if row["protocol_phase"] == "retention_s2c_cells"]
    cell_rows: list[dict[str, object]] = []
    for cell_index, (x_low_mm, y_low_mm) in enumerate(
        RETENTION_CELL_LOWER_LEFT_MM
    ):
        cell = [
            row for row in selected if int(row["spatial_stratum"]) == cell_index
        ]
        successes = sum(stage2_base._truth(row["success"]) for row in cell)
        total = len(cell)
        rate = successes / total if total else float("nan")
        lower, upper = stage2_base._wilson_interval(successes, total)
        cell_rows.append(
            {
                "cell_index": cell_index,
                "x_min_mm": x_low_mm,
                "x_max_mm": x_low_mm + RETENTION_CELL_SIZE_MM,
                "y_min_mm": y_low_mm,
                "y_max_mm": y_low_mm + RETENTION_CELL_SIZE_MM,
                "episodes": total,
                "successes": successes,
                "success_rate": rate,
                "wilson_95_lower": lower,
                "wilson_95_upper": upper,
                **_phase_counts(cell),
            }
        )
    csv_path = output_dir / "retention_cells.csv"
    stage2_base._write_dict_csv(csv_path, cell_rows)
    total_successes = sum(stage2_base._truth(row["success"]) for row in selected)
    expected = len(RETENTION_CELL_LOWER_LEFT_MM) * samples_per_cell
    return (
        {
            "rows": len(selected),
            "expected_rows": expected,
            "successes": total_successes,
            "success_rate": total_successes / len(selected) if selected else None,
            "minimum_cell_success_rate": min(
                (float(row["success_rate"]) for row in cell_rows),
                default=None,
            ),
            "outcomes": _phase_counts(selected),
            "complete": len(selected) == expected
            and all(int(row["episodes"]) == samples_per_cell for row in cell_rows),
        },
        [csv_path],
    )


def _summarize_sector(
    rows: list[dict[str, object]],
    spec: SectorSpec,
    samples_per_stratum: int,
    output_dir: Path,
) -> tuple[dict[str, object], list[Path]]:
    selected = [row for row in rows if row["protocol_phase"] == "active_polar_sector"]
    radius_min, radius_max = spec.radius_range_m
    angle_min, angle_max = spec.angle_range_deg
    radial_width = (radius_max - radius_min) / RADIAL_STRATA
    angular_width = (angle_max - angle_min) / ANGULAR_STRATA
    matrix = np.full((ANGULAR_STRATA, RADIAL_STRATA), np.nan)
    stratum_rows: list[dict[str, object]] = []
    for angular_index in range(ANGULAR_STRATA):
        for radial_index in range(RADIAL_STRATA):
            stratum = angular_index * RADIAL_STRATA + radial_index
            group = [
                row for row in selected if int(row["spatial_stratum"]) == stratum
            ]
            successes = sum(stage2_base._truth(row["success"]) for row in group)
            total = len(group)
            rate = successes / total if total else float("nan")
            matrix[angular_index, radial_index] = rate
            lower, upper = stage2_base._wilson_interval(successes, total)
            stratum_rows.append(
                {
                    "spatial_stratum": stratum,
                    "radial_stratum": radial_index,
                    "angular_stratum": angular_index,
                    "radius_min_mm": 1000.0
                    * (radius_min + radial_index * radial_width),
                    "radius_max_mm": 1000.0
                    * (radius_min + (radial_index + 1) * radial_width),
                    "angle_min_deg": angle_min + angular_index * angular_width,
                    "angle_max_deg": angle_min + (angular_index + 1) * angular_width,
                    "episodes": total,
                    "successes": successes,
                    "success_rate": rate,
                    "wilson_95_lower": lower,
                    "wilson_95_upper": upper,
                    **_phase_counts(group),
                }
            )
    csv_path = output_dir / "active_sector_strata.csv"
    stage2_base._write_dict_csv(csv_path, stratum_rows)
    figures = _make_polar_heatmap(matrix, spec, output_dir)
    total_successes = sum(stage2_base._truth(row["success"]) for row in selected)
    expected = RADIAL_STRATA * ANGULAR_STRATA * samples_per_stratum
    return (
        {
            "rows": len(selected),
            "expected_rows": expected,
            "successes": total_successes,
            "success_rate": total_successes / len(selected) if selected else None,
            "minimum_stratum_success_rate": min(
                (float(row["success_rate"]) for row in stratum_rows),
                default=None,
            ),
            "outcomes": _phase_counts(selected),
            "complete": len(selected) == expected
            and all(
                int(row["episodes"]) == samples_per_stratum for row in stratum_rows
            ),
        },
        [csv_path, *figures],
    )


def _summarize(
    rows: list[dict[str, object]],
    cfg: SectorEvaluateConfig,
    spec: SectorSpec,
    checkpoint_hash: str,
    output_dir: Path,
) -> tuple[dict[str, object], list[Path]]:
    nominal_repeats, retention_samples, sector_samples = _samples_for_mode(cfg.mode)
    nominal = [row for row in rows if row["protocol_phase"] == "nominal_repeat"]
    nominal_successes = sum(stage2_base._truth(row["success"]) for row in nominal)
    retention, retention_paths = _summarize_retention(
        rows, retention_samples, output_dir
    )
    sector, sector_paths = _summarize_sector(rows, spec, sector_samples, output_dir)

    raw_overlaps = [
        row
        for row in rows
        if stage2_base._truth(row["termination_overlap"])
    ]
    compatible_success_timeout = sum(
        stage2_base._truth(row["success"])
        and stage2_base._truth(row["timeout"])
        and not any(
            stage2_base._truth(row[name]) for name in ("fall", "oob", "nan")
        )
        for row in raw_overlaps
    )
    incompatible_overlaps = len(raw_overlaps) - compatible_success_timeout
    unknown = sum(str(row["outcome"]) == "unknown" for row in rows)
    pair_mismatches = sum(
        float(row["egg_pedestal_offset_pair_error_mm"]) > 1.0e-3 for row in rows
    )
    nan_count = sum(stage2_base._truth(row["nan"]) for row in rows)
    nan_rate = nan_count / len(rows) if rows else None
    nan_lower, nan_upper = stage2_base._wilson_interval(nan_count, len(rows))
    expected_rows = (
        nominal_repeats
        + len(RETENTION_CELL_LOWER_LEFT_MM) * retention_samples
        + RADIAL_STRATA * ANGULAR_STRATA * sector_samples
    )
    protocol_complete = (
        len(rows) == expected_rows
        and len(nominal) == nominal_repeats
        and bool(retention["complete"])
        and bool(sector["complete"])
        and pair_mismatches == 0
    )
    evidence_interpretable = (
        protocol_complete and unknown == 0 and incompatible_overlaps == 0
    )
    numerical_review_recommended = bool(
        nan_rate is not None and nan_rate >= NAN_RATE_REVIEW_TRIGGER
    )
    nominal_rate = nominal_successes / len(nominal) if nominal else None
    indicators = {
        "nominal_at_least_84_percent": bool(
            nominal_rate is not None and nominal_rate >= NOMINAL_SUCCESS_RATE_GUIDE
        ),
        "retention_at_least_90_percent": bool(
            retention["success_rate"] is not None
            and float(retention["success_rate"]) >= RETENTION_SUCCESS_RATE_GUIDE
        ),
        "active_sector_at_least_80_percent": bool(
            sector["success_rate"] is not None
            and float(sector["success_rate"]) >= SECTOR_SUCCESS_RATE_GUIDE
        ),
    }
    progression_recommended = (
        cfg.mode == "full"
        and evidence_interpretable
        and not numerical_review_recommended
        and all(indicators.values())
    )
    if cfg.mode == "smoke":
        next_action = "run_full_evaluation"
    elif not evidence_interpretable:
        next_action = "repair_or_explain_protocol_problem"
    elif numerical_review_recommended:
        next_action = "review_numerical_events_before_progression"
    elif not indicators["nominal_at_least_84_percent"]:
        next_action = "continue_same_arc_with_more_nominal_rehearsal"
    elif not indicators["retention_at_least_90_percent"]:
        next_action = "continue_same_arc_with_more_retained_region_rehearsal"
    elif not indicators["active_sector_at_least_80_percent"]:
        next_action = "continue_training_the_same_arc"
    elif spec.expansion_fraction < 1.0:
        next_action = "progress_to_next_arc"
    else:
        next_action = "run_dense_final_sector_map_then_prepare_hardware_transfer"

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": cfg.mode,
        "task_id": cfg.task_id,
        "stage_label": spec.label,
        "expansion_fraction": spec.expansion_fraction,
        "radius_range_mm": [1000.0 * value for value in spec.radius_range_m],
        "angle_range_deg": list(spec.angle_range_deg),
        "checkpoint_sha256": checkpoint_hash,
        "row_count": len(rows),
        "expected_rows": expected_rows,
        "nominal": {
            "rows": len(nominal),
            "successes": nominal_successes,
            "success_rate": nominal_rate,
            "outcomes": _phase_counts(nominal),
        },
        "retention": retention,
        "active_sector": sector,
        "all_outcomes": _phase_counts(rows),
        "termination_overlap_rows": len(raw_overlaps),
        "compatible_success_timeout_overlaps": compatible_success_timeout,
        "incompatible_physical_overlaps": incompatible_overlaps,
        "unknown_outcome_rows": unknown,
        "egg_pedestal_position_mismatches": pair_mismatches,
        "nan_events": nan_count,
        "nan_rate": nan_rate,
        "nan_rate_wilson_95": [nan_lower, nan_upper],
        "nan_rate_review_trigger": NAN_RATE_REVIEW_TRIGGER,
        "numerical_review_recommended": numerical_review_recommended,
        "protocol_complete": protocol_complete,
        "evidence_interpretable": evidence_interpretable,
        "progression_indicators": indicators,
        "progression_recommended": progression_recommended,
        "recommended_next_action": next_action,
        "decision_policy": {
            "nan": (
                "Count every NaN rollout as a failed episode and report its rate. "
                "A rate below 0.1% does not automatically veto progression."
            ),
            "success_timeout_overlap": (
                "A pure success+timeout boundary overlap is reported but is not "
                "treated as a physically incompatible outcome."
            ),
            "stage_navigation_guides": {
                "nominal_success_rate": NOMINAL_SUCCESS_RATE_GUIDE,
                "retention_success_rate": RETENTION_SUCCESS_RATE_GUIDE,
                "active_sector_success_rate": SECTOR_SUCCESS_RATE_GUIDE,
            },
        },
        "interpretation_note": (
            "The navigation guides decide whether to expand the curriculum, not "
            "whether the robot is hardware-ready. Full hardware readiness still "
            "requires a dense final-sector map and physical validation."
        ),
    }
    return summary, [*retention_paths, *sector_paths]


def run_evaluation(cfg: SectorEvaluateConfig) -> Path:
    if cfg.task_id not in SECTOR_SPECS:
        raise ValueError(f"Unknown sector task: {cfg.task_id}")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", cfg.training_git_commit):
        raise ValueError("--training-git-commit must be a full 40-character SHA")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", cfg.expected_checkpoint_sha256):
        raise ValueError("--expected-checkpoint-sha256 must contain 64 hex characters")
    if cfg.evaluation_seed != EVALUATION_SEED:
        raise ValueError(f"Sector protocol requires evaluation seed {EVALUATION_SEED}")
    if cfg.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")

    spec = SECTOR_SPECS[cfg.task_id]
    repo_root = Path(__file__).resolve().parents[1]
    evaluator_commit = stage2_base._run_git("rev-parse", "HEAD")
    dirty_output = stage2_base._run_git("status", "--porcelain")
    evaluator_dirty = bool(dirty_output)
    if evaluator_dirty and not cfg.allow_dirty_evaluator:
        raise RuntimeError(
            "Evaluator checkout is dirty. Commit the evaluator before running.\n"
            + dirty_output
        )

    checkpoint = Path(cfg.checkpoint_file).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    checkpoint_hash = stage2_base._sha256_file(checkpoint)
    if checkpoint_hash.lower() != cfg.expected_checkpoint_sha256.lower():
        raise RuntimeError(
            "Checkpoint hash does not match: "
            f"expected {cfg.expected_checkpoint_sha256}, observed {checkpoint_hash}"
        )

    output_dir = Path(cfg.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use a new directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_torch_backends(allow_tf32=True, deterministic=True)
    seed_rng(cfg.evaluation_seed, torch_deterministic=True)
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(cfg.task_id, play=False)
    agent_cfg = load_rl_cfg(cfg.task_id)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.auto_reset = False
    env_cfg.seed = cfg.evaluation_seed
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.observations["critic"].enable_corruption = False
    removed_event = env_cfg.events.pop("stage2_egg_pedestal_spawn", None)
    if removed_event is None:
        raise RuntimeError("Stage-2 spawn event was not found")

    task_hash = stage2_base._sha256_tree(repo_root / "src" / "spirob_mjlab")
    lock_path = repo_root / "uv.lock"
    lock_hash = (
        stage2_base._sha256_file(lock_path) if lock_path.is_file() else "missing"
    )
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    vec_env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    conditions = _evaluation_conditions(spec, cfg.mode)
    compatibility_cfg = stage2_base.Stage1EvaluateConfig(
        run_id=cfg.run_id,
        training_seed=cfg.training_seed,
        training_git_commit=cfg.training_git_commit,
        checkpoint_file=str(checkpoint),
        evaluation_seed=cfg.evaluation_seed,
        protocol="s2a-smoke",
        policy_mode="deterministic",
        output_dir=str(output_dir),
        device=device,
        s2_num_envs=cfg.num_envs,
    )
    identity: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": cfg.run_id,
        "training_seed": cfg.training_seed,
        "training_git_commit": cfg.training_git_commit,
        "evaluation_seed": cfg.evaluation_seed,
        "evaluator_git_commit": evaluator_commit,
        "evaluator_git_dirty": evaluator_dirty,
        "task_package_sha256": task_hash,
        "dependency_lock_sha256": lock_hash,
        "checkpoint_name": checkpoint.name,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "policy_mode": cfg.policy_mode,
    }

    csv_path = output_dir / "episodes.csv"
    rows: list[dict[str, object]] = []
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=stage2_base.CSV_COLUMNS)
            writer.writeheader()
            total_batches = (len(conditions) + cfg.num_envs - 1) // cfg.num_envs
            for batch_number, start in enumerate(
                range(0, len(conditions), cfg.num_envs), start=1
            ):
                batch_conditions = conditions[start : start + cfg.num_envs]
                batch_rows = stage2_base._run_stage2_batch(
                    vec_env,
                    policy,
                    compatibility_cfg,
                    batch_conditions,
                    identity,
                    start,
                )
                writer.writerows(batch_rows)
                stream.flush()
                rows.extend(batch_rows)
                successes = sum(
                    stage2_base._truth(row["success"]) for row in batch_rows
                )
                nan_count = sum(stage2_base._truth(row["nan"]) for row in batch_rows)
                print(
                    f"[sector {spec.label}] batch {batch_number}/{total_batches}: "
                    f"success {successes}/{len(batch_rows)}, NaN {nan_count}"
                )
    finally:
        vec_env.close()

    summary, supplemental_paths = _summarize(
        rows, cfg, spec, checkpoint_hash, output_dir
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    checkpoint_hash_path = output_dir / "checkpoint.sha256"
    checkpoint_hash_path.write_text(
        f"{checkpoint_hash}  {checkpoint.name}\n", encoding="utf-8"
    )
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": asdict(cfg),
        "sector_specification": asdict(spec),
        "evaluator_git_commit": evaluator_commit,
        "evaluator_git_dirty": evaluator_dirty,
        "evaluator_git_status": dirty_output.splitlines(),
        "task_package_sha256": task_hash,
        "dependency_lock_sha256": lock_hash,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "position_seeds": {
            "retention": RETENTION_POSITION_SEED,
            "sector": SECTOR_POSITION_SEED,
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.device(device))
            if device.startswith("cuda") and torch.cuda.is_available()
            else None
        ),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "mjlab",
                "mujoco",
                "mujoco-warp",
                "warp-lang",
                "rsl-rl-lib",
                "torch",
            )
        },
        "generated_files": [
            csv_path.name,
            summary_path.name,
            checkpoint_hash_path.name,
            *[
                path.relative_to(output_dir).as_posix()
                for path in supplemental_paths
            ],
        ],
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    stage2_base._write_manifest(output_dir)

    print(f"[sector {spec.label}] Episodes: {csv_path}")
    print(f"[sector {spec.label}] Summary: {summary_path}")
    print(f"[sector {spec.label}] Protocol complete: {summary['protocol_complete']}")
    print(
        f"[sector {spec.label}] Evidence interpretable: "
        f"{summary['evidence_interpretable']}"
    )
    print(f"[sector {spec.label}] NaN rate: {100.0 * float(summary['nan_rate']):.4f}%")
    print(f"[sector {spec.label}] Next action: {summary['recommended_next_action']}")
    return output_dir


def main() -> None:
    run_evaluation(tyro.cli(SectorEvaluateConfig))


if __name__ == "__main__":
    main()
