"""Compact frozen-policy promotion screen for SpiRob Stage 2-B.

This evaluator answers only the curriculum-promotion question:

* candidate: did the S2-B checkpoint retain +/-2 mm and acquire +/-5 mm?
* baseline: how does the S2-A checkpoint perform at the exact same +/-5 mm
  continuous positions?

The full candidate protocol contains 50 nominal repeats, 625 unique balanced
+/-2 mm positions, and 625 unique balanced +/-5 mm positions.  The full
baseline protocol contains the same 625 +/-5 mm positions used for the
candidate.  A smaller smoke mode validates the machinery before the full run.

Training configuration and reward code are not modified.  Spawn events are
removed during evaluation and paired egg/pedestal positions are imposed by the
existing frozen-policy batch evaluator.
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


PROTOCOL_VERSION = "S2-B-promotion-v1"
STAGE2A_TASK_ID = "Mjlab-SpiRob-EggToBucket-Stage2"
STAGE2B_TASK_ID = "Mjlab-SpiRob-EggToBucket-Stage2B"

EVALUATION_SEED = 2505
CORE_POSITION_SEED = 2502
EXPANDED_POSITION_SEED = 2505
STRATA_PER_AXIS = 5

FULL_NOMINAL_REPEATS = 50
FULL_SAMPLES_PER_STRATUM = 25
SMOKE_NOMINAL_REPEATS = 5
SMOKE_SAMPLES_PER_STRATUM = 2

NOMINAL_SUCCESS_MIN = 42
CORE_SUCCESS_RATE_MIN = 0.88
EXPANDED_SUCCESS_RATE_MIN = 0.85

Role = Literal["candidate", "baseline"]
Mode = Literal["smoke", "full"]


@dataclass(frozen=True)
class PromotionEvaluateConfig:
    """Inputs for one checkpoint in the paired S2-B promotion screen."""

    run_id: str
    """Scientific identifier for this evaluation."""

    role: Role
    """Candidate is S2-B; baseline is the earlier S2-A checkpoint."""

    training_seed: int
    """Seed used to train the checkpoint."""

    training_git_commit: str
    """Full Git commit used for checkpoint training."""

    checkpoint_file: str
    """Frozen local model_499.pt path."""

    expected_checkpoint_sha256: str
    """Previously recorded SHA-256; evaluation stops if it does not match."""

    output_dir: str
    """New evidence directory. It must be absent or empty."""

    mode: Mode = "smoke"
    """Run the small machinery check first; use full only after it passes."""

    evaluation_seed: int = EVALUATION_SEED
    """Fixed seed shared by candidate and baseline."""

    policy_mode: Literal["deterministic"] = "deterministic"
    """Use the policy mean action; no sampled policy action noise."""

    device: str | None = None
    """Defaults to cuda:0 when CUDA is available, otherwise cpu."""

    num_envs: int = 256
    """Parallel simulator slots."""

    allow_dirty_evaluator: bool = False
    """Diagnostic escape hatch; scientific runs must use a clean commit."""


def _balanced_conditions(
    *,
    range_mm: float,
    phase: str,
    samples_per_stratum: int,
    position_seed: int,
) -> list[stage2_base.Stage2Condition]:
    """Return unique, equal-allocation continuous positions for one square."""
    rng = np.random.default_rng(position_seed)
    cell_width_mm = 2.0 * range_mm / STRATA_PER_AXIS
    conditions: list[stage2_base.Stage2Condition] = []

    for stratum_y in range(STRATA_PER_AXIS):
        y_low = -range_mm + stratum_y * cell_width_mm
        y_high = y_low + cell_width_mm
        for stratum_x in range(STRATA_PER_AXIS):
            x_low = -range_mm + stratum_x * cell_width_mm
            x_high = x_low + cell_width_mm
            stratum = stratum_y * STRATA_PER_AXIS + stratum_x
            x_draws = rng.uniform(x_low, x_high, samples_per_stratum)
            y_draws = rng.uniform(y_low, y_high, samples_per_stratum)
            conditions.extend(
                stage2_base.Stage2Condition(
                    phase=phase,  # type: ignore[arg-type]
                    phase_episode_index=-1,
                    offset_x_mm=float(x_draws[index]),
                    offset_y_mm=float(y_draws[index]),
                    spatial_stratum=stratum,
                    stratum_x_index=stratum_x,
                    stratum_y_index=stratum_y,
                    repeat_index=index,
                    is_unique_continuous_spawn=True,
                )
                for index in range(samples_per_stratum)
            )

    offsets = {
        (condition.offset_x_mm, condition.offset_y_mm)
        for condition in conditions
    }
    if len(offsets) != len(conditions):
        raise RuntimeError(f"{phase} generator produced a duplicate offset")

    rng.shuffle(conditions)
    return [
        replace(condition, phase_episode_index=index)
        for index, condition in enumerate(conditions)
    ]


def _promotion_conditions(role: Role, mode: Mode) -> list[stage2_base.Stage2Condition]:
    samples_per_stratum = (
        FULL_SAMPLES_PER_STRATUM
        if mode == "full"
        else SMOKE_SAMPLES_PER_STRATUM
    )
    expanded = _balanced_conditions(
        range_mm=5.0,
        phase="promotion_expanded_5mm",
        samples_per_stratum=samples_per_stratum,
        position_seed=EXPANDED_POSITION_SEED,
    )
    if role == "baseline":
        return expanded

    nominal_repeats = (
        FULL_NOMINAL_REPEATS if mode == "full" else SMOKE_NOMINAL_REPEATS
    )
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
    core = _balanced_conditions(
        range_mm=2.0,
        phase="promotion_core_2mm",
        samples_per_stratum=samples_per_stratum,
        position_seed=CORE_POSITION_SEED,
    )
    return [*nominal, *core, *expanded]


def _phase_summary(
    rows: list[dict[str, object]],
    *,
    phase: str,
    range_mm: float,
    samples_per_stratum: int,
    output_dir: Path,
) -> tuple[dict[str, object], list[Path]]:
    selected = [row for row in rows if row["protocol_phase"] == phase]
    cell_width_mm = 2.0 * range_mm / STRATA_PER_AXIS
    matrix = np.full((STRATA_PER_AXIS, STRATA_PER_AXIS), np.nan)
    stratum_rows: list[dict[str, object]] = []
    stratum_successes: list[int] = []
    stratum_totals: list[int] = []

    for stratum in range(STRATA_PER_AXIS * STRATA_PER_AXIS):
        stratum_selected = [
            row for row in selected if int(row["spatial_stratum"]) == stratum
        ]
        successes = sum(stage2_base._truth(row["success"]) for row in stratum_selected)
        total = len(stratum_selected)
        rate = successes / total if total else float("nan")
        lower, upper = stage2_base._wilson_interval(successes, total)
        stratum_x = stratum % STRATA_PER_AXIS
        stratum_y = stratum // STRATA_PER_AXIS
        matrix[stratum_y, stratum_x] = rate
        stratum_successes.append(successes)
        stratum_totals.append(total)
        stratum_rows.append(
            {
                "phase": phase,
                "spatial_stratum": stratum,
                "stratum_x_index": stratum_x,
                "stratum_y_index": stratum_y,
                "x_min_mm": -range_mm + stratum_x * cell_width_mm,
                "x_max_mm": -range_mm + (stratum_x + 1) * cell_width_mm,
                "y_min_mm": -range_mm + stratum_y * cell_width_mm,
                "y_max_mm": -range_mm + (stratum_y + 1) * cell_width_mm,
                "episodes": total,
                "successes": successes,
                "falls": sum(
                    stage2_base._truth(row["fall"]) for row in stratum_selected
                ),
                "timeouts": sum(
                    stage2_base._truth(row["timeout"]) for row in stratum_selected
                ),
                "oob": sum(
                    stage2_base._truth(row["oob"]) for row in stratum_selected
                ),
                "nan": sum(
                    stage2_base._truth(row["nan"]) for row in stratum_selected
                ),
                "success_rate": rate,
                "wilson_95_lower": lower,
                "wilson_95_upper": upper,
            }
        )

    csv_path = output_dir / f"{phase}_strata_summary.csv"
    stage2_base._write_dict_csv(csv_path, stratum_rows)
    successes = sum(stratum_successes)
    total = sum(stratum_totals)
    lower, upper = stage2_base._stratified_interval(
        stratum_successes, stratum_totals
    )
    unique_positions = len(
        {
            (float(row["egg_offset_x_mm"]), float(row["egg_offset_y_mm"]))
            for row in selected
        }
    )
    expected = STRATA_PER_AXIS * STRATA_PER_AXIS * samples_per_stratum
    complete = (
        total == expected
        and unique_positions == expected
        and all(value == samples_per_stratum for value in stratum_totals)
    )

    ticks = tuple(
        -range_mm + (index + 0.5) * cell_width_mm
        for index in range(STRATA_PER_AXIS)
    )
    figure_paths = stage2_base._make_s2_heatmap(
        matrix,
        title=f"{phase.replace('_', ' ')} success by spatial stratum",
        note=(
            f"{samples_per_stratum} distinct continuous positions per equal-area "
            "stratum; promotion-screen evidence, not a deployment map."
        ),
        ticks=ticks,
        output_stem=output_dir / f"{phase}_success",
    )

    return (
        {
            "phase": phase,
            "range_mm": range_mm,
            "rows": total,
            "unique_positions": unique_positions,
            "successes": successes,
            "success_rate": successes / total if total else float("nan"),
            "stratified_95_lower": lower,
            "stratified_95_upper": upper,
            "minimum_stratum_success_rate": min(
                (
                    successes_in_stratum / total_in_stratum
                    for successes_in_stratum, total_in_stratum in zip(
                        stratum_successes, stratum_totals, strict=True
                    )
                    if total_in_stratum
                ),
                default=float("nan"),
            ),
            "samples_per_stratum": samples_per_stratum,
            "complete": complete,
            "outcomes": {
                name: sum(stage2_base._truth(row[name]) for row in selected)
                for name in ("success", "fall", "timeout", "oob", "nan")
            },
        },
        [csv_path, *figure_paths],
    )


def _make_summary(
    rows: list[dict[str, object]],
    cfg: PromotionEvaluateConfig,
    checkpoint_hash: str,
    output_dir: Path,
) -> tuple[dict[str, object], list[Path]]:
    samples_per_stratum = (
        FULL_SAMPLES_PER_STRATUM
        if cfg.mode == "full"
        else SMOKE_SAMPLES_PER_STRATUM
    )
    supplemental: list[Path] = []
    phase_summaries: dict[str, dict[str, object]] = {}

    if cfg.role == "candidate":
        core_summary, core_paths = _phase_summary(
            rows,
            phase="promotion_core_2mm",
            range_mm=2.0,
            samples_per_stratum=samples_per_stratum,
            output_dir=output_dir,
        )
        phase_summaries["core_2mm"] = core_summary
        supplemental.extend(core_paths)

    expanded_summary, expanded_paths = _phase_summary(
        rows,
        phase="promotion_expanded_5mm",
        range_mm=5.0,
        samples_per_stratum=samples_per_stratum,
        output_dir=output_dir,
    )
    phase_summaries["expanded_5mm"] = expanded_summary
    supplemental.extend(expanded_paths)

    nominal = [row for row in rows if row["protocol_phase"] == "nominal_repeat"]
    nominal_expected = (
        (FULL_NOMINAL_REPEATS if cfg.mode == "full" else SMOKE_NOMINAL_REPEATS)
        if cfg.role == "candidate"
        else 0
    )
    nominal_successes = sum(
        stage2_base._truth(row["success"]) for row in nominal
    )
    oob_count = sum(stage2_base._truth(row["oob"]) for row in rows)
    nan_count = sum(stage2_base._truth(row["nan"]) for row in rows)
    overlap_count = sum(
        stage2_base._truth(row["termination_overlap"]) for row in rows
    )
    unknown_count = sum(str(row["outcome"]) == "unknown" for row in rows)
    pair_error_max = max(
        (float(row["egg_pedestal_offset_pair_error_mm"]) for row in rows),
        default=float("nan"),
    )
    expected_rows = (
        nominal_expected
        + (STRATA_PER_AXIS**2 * samples_per_stratum if cfg.role == "candidate" else 0)
        + STRATA_PER_AXIS**2 * samples_per_stratum
    )
    complete = (
        len(rows) == expected_rows
        and len(nominal) == nominal_expected
        and all(bool(summary["complete"]) for summary in phase_summaries.values())
        and overlap_count == 0
        and unknown_count == 0
    )

    candidate_component_pass: bool | None = None
    if cfg.role == "candidate" and cfg.mode == "full" and complete:
        core = phase_summaries["core_2mm"]
        expanded = phase_summaries["expanded_5mm"]
        candidate_component_pass = (
            nominal_successes >= NOMINAL_SUCCESS_MIN
            and float(core["success_rate"]) >= CORE_SUCCESS_RATE_MIN
            and float(expanded["success_rate"]) >= EXPANDED_SUCCESS_RATE_MIN
            and oob_count == 0
            and nan_count == 0
            and overlap_count == 0
            and unknown_count == 0
        )

    return (
        {
            "protocol_version": PROTOCOL_VERSION,
            "role": cfg.role,
            "mode": cfg.mode,
            "checkpoint_sha256": checkpoint_hash,
            "row_count": len(rows),
            "nominal_rows": len(nominal),
            "nominal_successes": nominal_successes,
            "nominal_success_rate": (
                nominal_successes / len(nominal) if nominal else None
            ),
            "phases": phase_summaries,
            "oob_terminations_all_rows": oob_count,
            "nan_terminations_all_rows": nan_count,
            "termination_overlap_rows": overlap_count,
            "unknown_outcome_rows": unknown_count,
            "maximum_egg_pedestal_offset_pair_error_mm": pair_error_max,
            "protocol_complete": complete,
            "candidate_component_gate_applicable": (
                cfg.role == "candidate" and cfg.mode == "full" and complete
            ),
            "candidate_component_gate_pass": candidate_component_pass,
            "paired_promotion_gate_pass": None,
            "gate_definition": {
                "candidate_nominal_success_minimum": "42/50",
                "candidate_core_2mm_success_minimum": "88%",
                "candidate_expanded_5mm_success_minimum": "85%",
                "candidate_improvement_over_s2a_baseline_minimum": (
                    "5 percentage points; computed by summarizer"
                ),
                "oob": 0,
                "nan": 0,
                "termination_overlap": 0,
                "unknown_outcome": 0,
            },
            "interpretation_note": (
                "This is a curriculum-promotion screen. The balanced continuous "
                "positions estimate performance over the specified square, but "
                "25 samples per stratum are not a deployment-grade workspace map."
            ),
        },
        supplemental,
    )


def run_evaluation(cfg: PromotionEvaluateConfig) -> Path:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", cfg.training_git_commit):
        raise ValueError("--training-git-commit must be a full 40-character SHA")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", cfg.expected_checkpoint_sha256):
        raise ValueError("--expected-checkpoint-sha256 must contain 64 hex characters")
    if cfg.evaluation_seed != EVALUATION_SEED:
        raise ValueError(
            f"Promotion protocol requires evaluation seed {EVALUATION_SEED}"
        )
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
    task_id = STAGE2B_TASK_ID if cfg.role == "candidate" else STAGE2A_TASK_ID
    env_cfg = load_env_cfg(task_id, play=False)
    agent_cfg = load_rl_cfg(task_id)
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
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    conditions = _promotion_conditions(cfg.role, cfg.mode)
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
                    f"[S2-B promotion:{cfg.role}] batch "
                    f"{batch_number}/{total_batches}: "
                    f"success {successes}/{len(batch_rows)}"
                )
    finally:
        vec_env.close()

    summary, supplemental_paths = _make_summary(
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
        "task_id": task_id,
        "evaluator_git_commit": evaluator_commit,
        "evaluator_git_dirty": evaluator_dirty,
        "evaluator_git_status": dirty_output.splitlines(),
        "task_package_sha256": task_hash,
        "dependency_lock_sha256": lock_hash,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "position_seeds": {
            "core_2mm": CORE_POSITION_SEED,
            "paired_expanded_5mm": EXPANDED_POSITION_SEED,
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

    print(f"[S2-B promotion:{cfg.role}] Episodes: {csv_path}")
    print(f"[S2-B promotion:{cfg.role}] Summary: {summary_path}")
    print(f"[S2-B promotion:{cfg.role}] Checkpoint SHA-256: {checkpoint_hash}")
    print(
        f"[S2-B promotion:{cfg.role}] Protocol complete: "
        f"{summary['protocol_complete']}"
    )
    print(
        f"[S2-B promotion:{cfg.role}] Candidate component gate: "
        f"{summary['candidate_component_gate_pass']}"
    )
    return output_dir


def main() -> None:
    run_evaluation(tyro.cli(PromotionEvaluateConfig))


if __name__ == "__main__":
    main()
