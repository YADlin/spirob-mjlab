"""Build the final fixed-base workspace map for the frozen S2-C policy.

The full protocol evaluates an 11 x 11 exact-position grid spanning +/-10 mm
at 2 mm spacing, with 100 deterministic computational repeats per point.  It
does not train or alter the policy.  A smoke mode runs two repeats per point to
validate the machinery before the full 12,100-episode map.

The map is deliberately conservative.  A point is supported only when at least
90 of 100 rollouts succeed and none has an engineering-invalid outcome.  A
continuous grid cell is supported only when all four corner points are
supported.  All other cells are withheld from manipulation by the external
workspace gate.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
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


PROTOCOL_VERSION = "S2-C-workspace-map-v1"
TASK_ID = "Mjlab-SpiRob-EggToBucket-Stage2C"
EVALUATION_SEED = 2710
GRID_OFFSETS_MM = tuple(float(value) for value in range(-10, 11, 2))
FULL_REPEATS_PER_POINT = 100
SMOKE_REPEATS_PER_POINT = 2
SUPPORTED_SUCCESS_RATE_MIN = 0.90
UNSUPPORTED_SUCCESS_RATE_MAX = 0.10

Mode = Literal["smoke", "full"]


@dataclass(frozen=True)
class WorkspaceEvaluateConfig:
    """Inputs for the frozen S2-C workspace map."""

    run_id: str
    training_seed: int
    training_git_commit: str
    checkpoint_file: str
    expected_checkpoint_sha256: str
    output_dir: str
    mode: Mode = "smoke"
    evaluation_seed: int = EVALUATION_SEED
    policy_mode: Literal["deterministic"] = "deterministic"
    device: str | None = None
    num_envs: int = 256
    allow_dirty_evaluator: bool = False


def _workspace_conditions(mode: Mode) -> list[stage2_base.Stage2Condition]:
    repeats = (
        FULL_REPEATS_PER_POINT if mode == "full" else SMOKE_REPEATS_PER_POINT
    )
    conditions = [
        stage2_base.Stage2Condition(
            phase="exact_grid_repeat",
            phase_episode_index=-1,
            offset_x_mm=offset_x_mm,
            offset_y_mm=offset_y_mm,
            spatial_stratum=-1,
            stratum_x_index=x_index,
            stratum_y_index=y_index,
            repeat_index=repeat_index,
            is_unique_continuous_spawn=False,
        )
        for y_index, offset_y_mm in enumerate(GRID_OFFSETS_MM)
        for x_index, offset_x_mm in enumerate(GRID_OFFSETS_MM)
        for repeat_index in range(repeats)
    ]
    rng = np.random.default_rng(EVALUATION_SEED)
    rng.shuffle(conditions)
    return [
        replace(condition, phase_episode_index=index)
        for index, condition in enumerate(conditions)
    ]


def _point_classification(
    successes: int,
    total: int,
    engineering_failures: int,
    mode: Mode,
) -> str:
    if mode != "full":
        return "diagnostic_only"
    if engineering_failures:
        return "engineering_invalid"
    rate = successes / total
    if rate >= SUPPORTED_SUCCESS_RATE_MIN:
        return "supported"
    if rate < UNSUPPORTED_SUCCESS_RATE_MAX:
        return "unsupported"
    return "borderline"


def _summarize_workspace(
    rows: list[dict[str, object]],
    cfg: WorkspaceEvaluateConfig,
    checkpoint_hash: str,
    output_dir: Path,
) -> tuple[dict[str, object], list[Path]]:
    repeats = (
        FULL_REPEATS_PER_POINT
        if cfg.mode == "full"
        else SMOKE_REPEATS_PER_POINT
    )
    point_rows: list[dict[str, object]] = []
    point_by_index: dict[tuple[int, int], dict[str, object]] = {}
    success_matrix = np.full((len(GRID_OFFSETS_MM), len(GRID_OFFSETS_MM)), np.nan)

    for y_index, offset_y_mm in enumerate(GRID_OFFSETS_MM):
        for x_index, offset_x_mm in enumerate(GRID_OFFSETS_MM):
            selected = [
                row
                for row in rows
                if float(row["egg_offset_x_mm"]) == offset_x_mm
                and float(row["egg_offset_y_mm"]) == offset_y_mm
            ]
            successes = sum(stage2_base._truth(row["success"]) for row in selected)
            falls = sum(stage2_base._truth(row["fall"]) for row in selected)
            timeouts = sum(stage2_base._truth(row["timeout"]) for row in selected)
            oob = sum(stage2_base._truth(row["oob"]) for row in selected)
            nan = sum(stage2_base._truth(row["nan"]) for row in selected)
            overlaps = sum(
                stage2_base._truth(row["termination_overlap"]) for row in selected
            )
            unknown = sum(str(row["outcome"]) == "unknown" for row in selected)
            total = len(selected)
            rate = successes / total if total else float("nan")
            lower, upper = stage2_base._wilson_interval(successes, total)
            engineering_failures = oob + nan + overlaps + unknown
            classification = _point_classification(
                successes, total, engineering_failures, cfg.mode
            )
            success_matrix[y_index, x_index] = rate
            point = {
                "x_index": x_index,
                "y_index": y_index,
                "offset_x_mm": offset_x_mm,
                "offset_y_mm": offset_y_mm,
                "computational_repeats": total,
                "successes": successes,
                "falls": falls,
                "timeouts": timeouts,
                "oob": oob,
                "nan": nan,
                "termination_overlaps": overlaps,
                "unknown_outcomes": unknown,
                "success_rate": rate,
                "wilson_95_lower": lower,
                "wilson_95_upper": upper,
                "classification": classification,
            }
            point_rows.append(point)
            point_by_index[(x_index, y_index)] = point

    point_csv = output_dir / "workspace_points.csv"
    stage2_base._write_dict_csv(point_csv, point_rows)

    cell_rows: list[dict[str, object]] = []
    cell_matrix = np.full((len(GRID_OFFSETS_MM) - 1, len(GRID_OFFSETS_MM) - 1), np.nan)
    for y_index in range(len(GRID_OFFSETS_MM) - 1):
        for x_index in range(len(GRID_OFFSETS_MM) - 1):
            corners = [
                point_by_index[(x_index, y_index)],
                point_by_index[(x_index + 1, y_index)],
                point_by_index[(x_index, y_index + 1)],
                point_by_index[(x_index + 1, y_index + 1)],
            ]
            supported = (
                cfg.mode == "full"
                and all(point["classification"] == "supported" for point in corners)
            )
            cell_matrix[y_index, x_index] = 1.0 if supported else 0.0
            cell_rows.append(
                {
                    "cell_x_index": x_index,
                    "cell_y_index": y_index,
                    "x_min_mm": GRID_OFFSETS_MM[x_index],
                    "x_max_mm": GRID_OFFSETS_MM[x_index + 1],
                    "y_min_mm": GRID_OFFSETS_MM[y_index],
                    "y_max_mm": GRID_OFFSETS_MM[y_index + 1],
                    "minimum_corner_success_rate": min(
                        float(point["success_rate"]) for point in corners
                    ),
                    "all_four_corners_supported": supported,
                    "deployment_decision": (
                        "MANIPULATION_ALLOWED"
                        if supported
                        else "OUTSIDE_DEMONSTRATED_WORKSPACE"
                    ),
                }
            )

    cell_csv = output_dir / "workspace_cells.csv"
    stage2_base._write_dict_csv(cell_csv, cell_rows)

    point_figures = stage2_base._make_s2_heatmap(
        success_matrix,
        title="S2-C exact-position computational success",
        note=(
            f"{repeats} deterministic computational repeats per point; "
            "supported means at least 90/100 with no engineering-invalid outcome."
        ),
        ticks=GRID_OFFSETS_MM,
        output_stem=output_dir / "workspace_point_success",
    )
    cell_ticks = tuple(
        (GRID_OFFSETS_MM[index] + GRID_OFFSETS_MM[index + 1]) / 2.0
        for index in range(len(GRID_OFFSETS_MM) - 1)
    )
    cell_figures = stage2_base._make_s2_heatmap(
        cell_matrix,
        title="Conservative S2-C manipulation cells",
        note=(
            "A cell is allowed only when all four evaluated corner points are "
            "supported; 1=allowed and 0=decline."
        ),
        ticks=cell_ticks,
        output_stem=output_dir / "workspace_supported_cells",
    )

    expected_rows = len(GRID_OFFSETS_MM) ** 2 * repeats
    complete = (
        len(rows) == expected_rows
        and all(int(point["computational_repeats"]) == repeats for point in point_rows)
    )
    engineering = {
        "oob": sum(stage2_base._truth(row["oob"]) for row in rows),
        "nan": sum(stage2_base._truth(row["nan"]) for row in rows),
        "termination_overlap": sum(
            stage2_base._truth(row["termination_overlap"]) for row in rows
        ),
        "unknown_outcome": sum(str(row["outcome"]) == "unknown" for row in rows),
    }
    map_ready = None
    if cfg.mode == "full":
        map_ready = complete and all(value == 0 for value in engineering.values())

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": cfg.mode,
        "checkpoint_sha256": checkpoint_hash,
        "grid_offsets_mm": list(GRID_OFFSETS_MM),
        "grid_points": len(point_rows),
        "repeats_per_point": repeats,
        "row_count": len(rows),
        "successes": sum(stage2_base._truth(row["success"]) for row in rows),
        "overall_computational_success_rate": (
            sum(stage2_base._truth(row["success"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "supported_points": sum(
            point["classification"] == "supported" for point in point_rows
        ),
        "borderline_points": sum(
            point["classification"] == "borderline" for point in point_rows
        ),
        "unsupported_points": sum(
            point["classification"] == "unsupported" for point in point_rows
        ),
        "engineering_invalid_points": sum(
            point["classification"] == "engineering_invalid" for point in point_rows
        ),
        "supported_cells": sum(
            bool(cell["all_four_corners_supported"]) for cell in cell_rows
        ),
        "total_cells": len(cell_rows),
        "engineering_outcomes": engineering,
        "protocol_complete": complete,
        "workspace_map_ready": map_ready,
        "classification_definition": {
            "supported": ">=90/100 successes and zero engineering-invalid outcomes",
            "borderline": "10-89/100 successes",
            "unsupported": "<10/100 successes",
            "allowed_cell": "all four corner points supported",
        },
        "deployment_rule": (
            "Attempt manipulation only inside cells marked MANIPULATION_ALLOWED; "
            "otherwise return OUTSIDE_DEMONSTRATED_WORKSPACE."
        ),
        "interpretation_note": (
            "Repeated deterministic rollouts quantify computational repeatability "
            "under MuJoCo Warp, not physical-world success probability. A weak "
            "point is unsupported by this controller, not proven unreachable."
        ),
    }
    return summary, [point_csv, cell_csv, *point_figures, *cell_figures]


def run_evaluation(cfg: WorkspaceEvaluateConfig) -> Path:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", cfg.training_git_commit):
        raise ValueError("--training-git-commit must be a full 40-character SHA")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", cfg.expected_checkpoint_sha256):
        raise ValueError("--expected-checkpoint-sha256 must contain 64 hex characters")
    if cfg.evaluation_seed != EVALUATION_SEED:
        raise ValueError(f"Workspace protocol requires evaluation seed {EVALUATION_SEED}")
    if cfg.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")

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
            "Checkpoint hash does not match the frozen evidence. "
            f"Expected {cfg.expected_checkpoint_sha256}, observed {checkpoint_hash}"
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
    env_cfg = load_env_cfg(TASK_ID, play=False)
    agent_cfg = load_rl_cfg(TASK_ID)
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
    lock_hash = stage2_base._sha256_file(lock_path) if lock_path.is_file() else "missing"

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    vec_env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    conditions = _workspace_conditions(cfg.mode)
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
                print(
                    f"[S2-C workspace] batch {batch_number}/{total_batches}: "
                    f"success {successes}/{len(batch_rows)}"
                )
    finally:
        vec_env.close()

    summary, supplemental_paths = _summarize_workspace(
        rows, cfg, checkpoint_hash, output_dir
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
        "task_id": TASK_ID,
        "evaluator_git_commit": evaluator_commit,
        "evaluator_git_dirty": evaluator_dirty,
        "evaluator_git_status": dirty_output.splitlines(),
        "task_package_sha256": task_hash,
        "dependency_lock_sha256": lock_hash,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
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
        "known_reproducibility_limit": (
            "Deterministic means mean policy actions. MuJoCo Warp may still "
            "produce different outcomes from repeated identical conditions."
        ),
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

    print(f"[S2-C workspace] Episodes: {csv_path}")
    print(f"[S2-C workspace] Summary: {summary_path}")
    print(f"[S2-C workspace] Checkpoint SHA-256: {checkpoint_hash}")
    print(f"[S2-C workspace] Protocol complete: {summary['protocol_complete']}")
    print(f"[S2-C workspace] Map ready: {summary['workspace_map_ready']}")
    return output_dir


def main() -> None:
    run_evaluation(tyro.cli(WorkspaceEvaluateConfig))


if __name__ == "__main__":
    main()
