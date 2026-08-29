"""Episode-level evaluator for frozen SpiRob S1 and S2-A policies.

The original S1 protocol is preserved.  The S2-A protocol evaluates a frozen
policy with three complementary checks:

1. 100 deterministic repeats at the nominal physical condition;
2. 10,000 distinct, balanced continuous spawns in the trained +/-2 mm square;
3. 100 deterministic repeats at each point of a 5 x 5 exact-position grid.

The continuous spawns estimate performance over a defined spatial sampling
distribution.  Exact-condition repeats measure computational repeatability;
they are not independent physical trials.  Every result is written at episode
level with checkpoint and source provenance.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
import tyro
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.random import seed_rng
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

import mjlab.tasks  # noqa: F401  # Populate MJLab's built-in registry.
import spirob_mjlab  # noqa: F401  # Register the SpiRob task.


STAGE1_TASK_ID = "Mjlab-SpiRob-EggToBucket-Stage1"
STAGE2_TASK_ID = "Mjlab-SpiRob-EggToBucket-Stage2"
S1_PROTOCOL_VERSION = "S1-v1"
S2A_PROTOCOL_VERSION = "S2-A-v1"
S1_EVALUATION_SEED = 1001
S1_GRID_OFFSETS_MM = (-10, -5, 0, 5, 10)
S2A_EVALUATION_SEED = 2002
S2A_RANGE_MM = 2.0
S2A_STRATA_PER_AXIS = 5
S2A_NOMINAL_REPEATS = 100
S2A_CONTINUOUS_SAMPLES_PER_STRATUM = 400
S2A_GRID_REPEATS_PER_POINT = 100
S2A_GRID_OFFSETS_MM = (-2.0, -1.0, 0.0, 1.0, 2.0)

S2A_SMOKE_NOMINAL_REPEATS = 5
S2A_SMOKE_CONTINUOUS_SAMPLES_PER_STRATUM = 10
S2A_SMOKE_GRID_REPEATS_PER_POINT = 1

S2A_ROBOT_CENTERLINE_X_M = 0.0
S2A_MIN_ROBOT_CENTERLINE_CLEARANCE_M = 0.038
S2A_BUCKET_XY_LOCAL_M = (-0.05, 0.15)
S2A_MIN_BUCKET_CENTER_CLEARANCE_M = 0.055

S2A_NOMINAL_SUCCESS_MIN = 90
S2A_CONTINUOUS_SUCCESS_RATE_MIN = 0.90
S2A_STRATUM_SUCCESS_RATE_MIN = 0.80


CSV_COLUMNS = (
    # Identity and provenance.
    "protocol_version",
    "evaluated_at_utc",
    "run_id",
    "training_seed",
    "training_git_commit",
    "evaluation_seed",
    "evaluator_git_commit",
    "evaluator_git_dirty",
    "task_package_sha256",
    "dependency_lock_sha256",
    "checkpoint_name",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
    "policy_mode",
    # Episode and physical-condition identity.
    "episode_index",
    "protocol_phase",
    "phase_episode_index",
    "physical_condition_id",
    "spatial_stratum",
    "stratum_x_index",
    "stratum_y_index",
    "condition_repeat_index",
    "is_unique_continuous_spawn",
    # Initial condition (environment-local coordinates).
    "egg_offset_x_mm",
    "egg_offset_y_mm",
    "actual_egg_offset_x_mm",
    "actual_egg_offset_y_mm",
    "actual_pedestal_offset_x_mm",
    "actual_pedestal_offset_y_mm",
    "egg_pedestal_offset_pair_error_mm",
    "initial_egg_x_m",
    "initial_egg_y_m",
    "initial_egg_z_m",
    "initial_bucket_site_x_m",
    "initial_bucket_site_y_m",
    "initial_bucket_site_z_m",
    "initial_base_x_m",
    "initial_base_y_m",
    "initial_base_z_m",
    "initial_tendon_0_length_m",
    "initial_tendon_1_length_m",
    "initial_tendon_0_velocity_m_s",
    "initial_tendon_1_velocity_m_s",
    "initial_tendon_0_command_m",
    "initial_tendon_1_command_m",
    # Outcome.
    "outcome",
    "success",
    "fall",
    "timeout",
    "oob",
    "nan",
    "termination_overlap",
    "episode_steps",
    "episode_length_s",
    "return",
    # Diagnostics.
    "min_egg_bucket_xy_distance_m",
    "min_egg_bucket_3d_distance_m",
    "max_egg_z_m",
    "final_egg_x_m",
    "final_egg_y_m",
    "final_egg_z_m",
    "final_bucket_site_x_m",
    "final_bucket_site_y_m",
    "final_bucket_site_z_m",
    "final_base_x_m",
    "final_base_y_m",
    "final_base_z_m",
    "final_tendon_0_length_m",
    "final_tendon_1_length_m",
    "final_tendon_0_command_m",
    "final_tendon_1_command_m",
    "trajectory_fingerprint_sha256",
    "video_file",
)


@dataclass(frozen=True)
class Stage1EvaluateConfig:
    """Configuration for one frozen-checkpoint S1 or S2-A evaluation."""

    run_id: str
    """Scientific run ID, for example R002, S1-S43, or S1-S44."""

    training_seed: int
    """Seed used to train this checkpoint."""

    training_git_commit: str
    """Git commit from which this checkpoint was trained."""

    checkpoint_file: str | None = None
    """Local checkpoint. Mutually exclusive with wandb_run_path."""

    wandb_run_path: str | None = None
    """W&B path ENTITY/PROJECT/RUN_ID. Mutually exclusive with checkpoint_file."""

    wandb_checkpoint_name: str | None = None
    """Exact checkpoint filename in W&B, for example model_499.pt."""

    evaluation_seed: int | None = None
    """Defaults to the preregistered seed for the selected protocol."""

    protocol: Literal[
        "nominal",
        "grid",
        "s1",
        "s2a-smoke",
        "s2a",
    ] = "s1"
    """S1 is unchanged; S2-A adds balanced continuous and exact-grid tests."""

    policy_mode: Literal["deterministic", "stochastic"] = "deterministic"
    """S1 requires deterministic; stochastic mode is diagnostic only."""

    output_dir: str = "outputs/s1/evaluation"
    """New directory for CSV, metadata, hashes, video, and figures."""

    device: str | None = None
    """Defaults to cuda:0 when CUDA is available, otherwise cpu."""

    log_root: str = "logs/rsl_rl"
    """Checkpoint cache root used by MJLab for W&B downloads."""

    record_video: bool = False
    """Record the first nominal episode. Required evidence for the S1 gate."""

    allow_dirty_evaluator: bool = False
    """Allow an evaluation from a dirty checkout while recording that fact."""

    s2_num_envs: int = 256
    """Parallel environments used only by S2-A protocols."""


@dataclass(frozen=True)
class Condition:
    phase: Literal["nominal_repeat", "grid"]
    phase_episode_index: int
    offset_x_mm: int
    offset_y_mm: int

    @property
    def physical_condition_id(self) -> str:
        return (
            f"egg_dx{self.offset_x_mm:+04d}mm_"
            f"dy{self.offset_y_mm:+04d}mm"
        )


@dataclass(frozen=True)
class Stage2Condition:
    phase: Literal[
        "nominal_repeat",
        "continuous_stratified",
        "exact_grid_repeat",
    ]
    phase_episode_index: int
    offset_x_mm: float
    offset_y_mm: float
    spatial_stratum: int
    stratum_x_index: int
    stratum_y_index: int
    repeat_index: int
    is_unique_continuous_spawn: bool

    @property
    def physical_condition_id(self) -> str:
        if self.phase == "nominal_repeat":
            return "egg_dx+0.000000mm_dy+0.000000mm"
        if self.phase == "exact_grid_repeat":
            return (
                f"egg_dx{self.offset_x_mm:+.3f}mm_"
                f"dy{self.offset_y_mm:+.3f}mm"
            )
        return (
            f"continuous_{self.phase_episode_index:05d}_"
            f"dx{self.offset_x_mm:+.9f}mm_"
            f"dy{self.offset_y_mm:+.9f}mm"
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=_repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> str:
    """Hash relative paths and bytes so renames also change the digest."""
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"


def _conditions(protocol: str) -> list[Condition]:
    conditions: list[Condition] = []
    if protocol in {"nominal", "s1"}:
        conditions.extend(
            Condition("nominal_repeat", repeat, 0, 0) for repeat in range(5)
        )
    if protocol in {"grid", "s1"}:
        conditions.extend(
            Condition("grid", index, dx, dy)
            for index, (dy, dx) in enumerate(
                (dy, dx)
                for dy in S1_GRID_OFFSETS_MM
                for dx in S1_GRID_OFFSETS_MM
            )
        )
    return conditions


def _stage2_conditions(
    protocol: Literal["s2a-smoke", "s2a"],
    evaluation_seed: int,
) -> list[Stage2Condition]:
    """Build the complete preregistered S2-A condition list.

    Continuous points are independent uniform draws within each equal-area
    stratum.  Equal allocation makes the overall success fraction the
    stratified estimate for a uniform position distribution over the square.
    """
    if protocol == "s2a-smoke":
        nominal_repeats = S2A_SMOKE_NOMINAL_REPEATS
        continuous_per_stratum = S2A_SMOKE_CONTINUOUS_SAMPLES_PER_STRATUM
        grid_repeats = S2A_SMOKE_GRID_REPEATS_PER_POINT
    else:
        nominal_repeats = S2A_NOMINAL_REPEATS
        continuous_per_stratum = S2A_CONTINUOUS_SAMPLES_PER_STRATUM
        grid_repeats = S2A_GRID_REPEATS_PER_POINT

    rng = np.random.default_rng(evaluation_seed)
    conditions: list[Stage2Condition] = [
        Stage2Condition(
            phase="nominal_repeat",
            phase_episode_index=repeat_index,
            offset_x_mm=0.0,
            offset_y_mm=0.0,
            spatial_stratum=-1,
            stratum_x_index=-1,
            stratum_y_index=-1,
            repeat_index=repeat_index,
            is_unique_continuous_spawn=False,
        )
        for repeat_index in range(nominal_repeats)
    ]

    cell_width_mm = 2.0 * S2A_RANGE_MM / S2A_STRATA_PER_AXIS
    continuous: list[Stage2Condition] = []
    for stratum_y in range(S2A_STRATA_PER_AXIS):
        y_low = -S2A_RANGE_MM + stratum_y * cell_width_mm
        y_high = y_low + cell_width_mm
        for stratum_x in range(S2A_STRATA_PER_AXIS):
            x_low = -S2A_RANGE_MM + stratum_x * cell_width_mm
            x_high = x_low + cell_width_mm
            stratum = stratum_y * S2A_STRATA_PER_AXIS + stratum_x
            x_draws = rng.uniform(x_low, x_high, continuous_per_stratum)
            y_draws = rng.uniform(y_low, y_high, continuous_per_stratum)
            continuous.extend(
                Stage2Condition(
                    phase="continuous_stratified",
                    phase_episode_index=-1,
                    offset_x_mm=float(x_draws[index]),
                    offset_y_mm=float(y_draws[index]),
                    spatial_stratum=stratum,
                    stratum_x_index=stratum_x,
                    stratum_y_index=stratum_y,
                    repeat_index=index,
                    is_unique_continuous_spawn=True,
                )
                for index in range(continuous_per_stratum)
            )

    unique_offsets = {
        (condition.offset_x_mm, condition.offset_y_mm)
        for condition in continuous
    }
    if len(unique_offsets) != len(continuous):
        raise RuntimeError("Continuous S2-A generator produced a duplicate offset")
    rng.shuffle(continuous)
    conditions.extend(
        replace(condition, phase_episode_index=index)
        for index, condition in enumerate(continuous)
    )

    exact_grid: list[Stage2Condition] = []
    for offset_y_mm in S2A_GRID_OFFSETS_MM:
        for offset_x_mm in S2A_GRID_OFFSETS_MM:
            if offset_x_mm == 0.0 and offset_y_mm == 0.0:
                # The nominal repeats are reused as the grid centre.
                continue
            exact_grid.extend(
                Stage2Condition(
                    phase="exact_grid_repeat",
                    phase_episode_index=-1,
                    offset_x_mm=offset_x_mm,
                    offset_y_mm=offset_y_mm,
                    spatial_stratum=-1,
                    stratum_x_index=-1,
                    stratum_y_index=-1,
                    repeat_index=repeat_index,
                    is_unique_continuous_spawn=False,
                )
                for repeat_index in range(grid_repeats)
            )
    rng.shuffle(exact_grid)
    conditions.extend(
        replace(condition, phase_episode_index=index)
        for index, condition in enumerate(exact_grid)
    )
    return conditions


def _resolve_checkpoint(
    cfg: Stage1EvaluateConfig,
    experiment_name: str,
) -> Path:
    has_local = cfg.checkpoint_file is not None
    has_wandb = cfg.wandb_run_path is not None
    if has_local == has_wandb:
        raise ValueError(
            "Provide exactly one of --checkpoint-file or --wandb-run-path."
        )

    if cfg.checkpoint_file is not None:
        path = Path(cfg.checkpoint_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    assert cfg.wandb_run_path is not None
    log_root = (Path(cfg.log_root) / experiment_name).resolve()
    path, _ = get_wandb_checkpoint_path(
        log_root,
        Path(cfg.wandb_run_path),
        cfg.wandb_checkpoint_name,
    )
    return path.resolve()


def _local_position(raw_env: ManagerBasedRlEnv, world_position: torch.Tensor) -> np.ndarray:
    local = world_position[0] - raw_env.scene.env_origins[0]
    return local.detach().cpu().numpy().astype(np.float64)


def _snapshot(raw_env: ManagerBasedRlEnv) -> dict[str, np.ndarray]:
    egg = raw_env.scene["egg"]
    bucket = raw_env.scene["bucket"]
    robot = raw_env.scene["robot"]

    bucket_site_ids, _ = bucket.find_sites(("bucket_site",), preserve_order=True)
    tendon_ids, _ = robot.find_tendons(
        ("cable_0", "cable_1"), preserve_order=True
    )

    action_term = raw_env.action_manager.get_term("cable_len")
    command = getattr(action_term, "rate_limited_action")

    return {
        "egg": _local_position(raw_env, egg.data.root_link_pos_w),
        "bucket": _local_position(
            raw_env,
            bucket.data.site_pos_w[:, bucket_site_ids].squeeze(1),
        ),
        "base": _local_position(raw_env, robot.data.root_link_pos_w),
        "tendon_len": (
            robot.data.tendon_len[:, tendon_ids][0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        ),
        "tendon_vel": (
            robot.data.tendon_vel[:, tendon_ids][0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        ),
        "tendon_command": (
            command[0].detach().cpu().numpy().astype(np.float64)
        ),
    }


def _snapshot_batch(raw_env: ManagerBasedRlEnv) -> dict[str, np.ndarray]:
    """Return local-frame state arrays for every parallel environment."""
    egg = raw_env.scene["egg"]
    pedestal = raw_env.scene["pedestal"]
    bucket = raw_env.scene["bucket"]
    robot = raw_env.scene["robot"]

    bucket_site_ids, _ = bucket.find_sites(("bucket_site",), preserve_order=True)
    tendon_ids, _ = robot.find_tendons(
        ("cable_0", "cable_1"), preserve_order=True
    )
    origins = raw_env.scene.env_origins.detach()
    action_term = raw_env.action_manager.get_term("cable_len")
    command = getattr(action_term, "rate_limited_action")

    def local(values: torch.Tensor) -> np.ndarray:
        return (
            (values - origins)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    return {
        "egg": local(egg.data.root_link_pos_w),
        "pedestal": local(pedestal.data.root_link_pos_w),
        "bucket": local(
            bucket.data.site_pos_w[:, bucket_site_ids].squeeze(1)
        ),
        "base": local(robot.data.root_link_pos_w),
        "tendon_len": (
            robot.data.tendon_len[:, tendon_ids]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        ),
        "tendon_vel": (
            robot.data.tendon_vel[:, tendon_ids]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        ),
        "tendon_command": (
            command.detach().cpu().numpy().astype(np.float64)
        ),
    }


def _reset_s2_batch(
    vec_env: RslRlVecEnvWrapper,
    conditions: Sequence[Stage2Condition],
    evaluation_seed: int,
) -> tuple[
    TensorDict,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """Reset a parallel batch and impose paired egg/pedestal XY offsets."""
    raw_env = vec_env.unwrapped
    if len(conditions) > raw_env.num_envs:
        raise ValueError("Condition batch is larger than the environment batch")

    raw_env.reset(seed=evaluation_seed)
    env_ids = torch.arange(
        raw_env.num_envs,
        dtype=torch.int64,
        device=raw_env.device,
    )
    offsets = torch.zeros((raw_env.num_envs, 2), device=raw_env.device)
    requested = torch.tensor(
        [(condition.offset_x_mm, condition.offset_y_mm) for condition in conditions],
        dtype=torch.float32,
        device=raw_env.device,
    ) / 1000.0
    offsets[: len(conditions)] = requested

    egg = raw_env.scene["egg"]
    egg_state = egg.data.default_root_state[env_ids].clone()
    egg_state[:, :3] += raw_env.scene.env_origins[env_ids]
    egg_state[:, :2] += offsets
    egg.write_root_state_to_sim(egg_state, env_ids=env_ids)

    pedestal = raw_env.scene["pedestal"]
    if not pedestal.is_mocap:
        raise RuntimeError("S2-A evaluation requires the movable pedestal")
    default_pedestal_state = pedestal.data.default_root_state[env_ids]
    pedestal_pose = torch.empty((raw_env.num_envs, 7), device=raw_env.device)
    pedestal_pose[:, :3] = (
        default_pedestal_state[:, :3] + raw_env.scene.env_origins[env_ids]
    )
    pedestal_pose[:, :2] += offsets
    pedestal_pose[:, 3:7] = default_pedestal_state[:, 3:7]
    pedestal.write_mocap_pose_to_sim(pedestal_pose, env_ids=env_ids)

    raw_env.sim.forward()
    raw_env.sim.sense()
    obs_dict = raw_env.observation_manager.compute(update_history=True)
    obs = TensorDict(obs_dict, batch_size=[raw_env.num_envs])

    snap = _snapshot_batch(raw_env)
    initial_xy_distance = np.linalg.norm(
        snap["egg"][:, :2] - snap["bucket"][:, :2], axis=1
    )
    setattr(
        raw_env,
        "_stage1_previous_egg_bucket_distance",
        torch.as_tensor(
            initial_xy_distance,
            dtype=torch.float32,
            device=raw_env.device,
        ),
    )

    default_egg_xy = (
        egg.data.default_root_state[:, :2].detach().cpu().numpy().astype(np.float64)
    )
    default_pedestal_xy = (
        pedestal.data.default_root_state[:, :2]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    actual_egg_offsets = snap["egg"][:, :2] - default_egg_xy
    actual_pedestal_offsets = snap["pedestal"][:, :2] - default_pedestal_xy

    active = slice(0, len(conditions))
    requested_np = offsets[active].detach().cpu().numpy().astype(np.float64)
    applied_error = np.max(
        np.abs(actual_egg_offsets[active] - requested_np), initial=0.0
    )
    pair_error = np.max(
        np.abs(
            actual_egg_offsets[active] - actual_pedestal_offsets[active]
        ),
        initial=0.0,
    )
    if applied_error > 1.0e-6:
        raise RuntimeError(
            f"S2-A requested/applied spawn error is {applied_error * 1000.0:.6f} mm"
        )
    if pair_error > 1.0e-6:
        raise RuntimeError(
            f"S2-A egg/pedestal pair error is {pair_error * 1000.0:.6f} mm"
        )

    egg_xy = snap["egg"][active, :2]
    robot_clear = (
        np.abs(egg_xy[:, 0] - S2A_ROBOT_CENTERLINE_X_M)
        >= S2A_MIN_ROBOT_CENTERLINE_CLEARANCE_M - 1.0e-6
    )
    bucket_clear = (
        np.linalg.norm(
            egg_xy - np.asarray(S2A_BUCKET_XY_LOCAL_M, dtype=np.float64),
            axis=1,
        )
        >= S2A_MIN_BUCKET_CENTER_CLEARANCE_M - 1.0e-6
    )
    if not np.all(robot_clear & bucket_clear):
        raise RuntimeError("S2-A evaluator generated a collision-screen-invalid spawn")
    if not all(np.isfinite(values[active]).all() for values in snap.values()):
        raise RuntimeError("S2-A evaluator generated a non-finite initial state")

    return obs, snap, actual_egg_offsets, actual_pedestal_offsets


def _reset_to_condition(
    vec_env: RslRlVecEnvWrapper,
    condition: Condition,
    evaluation_seed: int,
) -> tuple[TensorDict, dict[str, np.ndarray]]:
    """Reset exactly, then apply the evaluation-only egg XY offset."""
    raw_env = vec_env.unwrapped
    raw_env.reset(seed=evaluation_seed)

    egg = raw_env.scene["egg"]
    env_ids = torch.tensor([0], dtype=torch.int64, device=raw_env.device)
    root_state = egg.data.default_root_state[env_ids].clone()
    root_state[:, 0:3] += raw_env.scene.env_origins[env_ids]
    root_state[:, 0] += condition.offset_x_mm / 1000.0
    root_state[:, 1] += condition.offset_y_mm / 1000.0
    egg.write_root_state_to_sim(root_state, env_ids=env_ids)

    # Refresh all derived quantities after the direct reset-state write.
    raw_env.sim.forward()
    raw_env.sim.sense()
    obs_dict = raw_env.observation_manager.compute(update_history=True)
    obs = TensorDict(obs_dict, batch_size=[raw_env.num_envs])

    # Explicitly align the progress-reward memory with the imposed initial state.
    snap = _snapshot(raw_env)
    initial_xy_distance = float(
        np.linalg.norm(snap["egg"][:2] - snap["bucket"][:2])
    )
    setattr(
        raw_env,
        "_stage1_previous_egg_bucket_distance",
        torch.tensor([initial_xy_distance], device=raw_env.device),
    )
    return obs, snap


def _finite_min(previous: float, value: float) -> float:
    return min(previous, value) if np.isfinite(value) else previous


def _finite_max(previous: float, value: float) -> float:
    return max(previous, value) if np.isfinite(value) else previous


def _termination_flags(raw_env: ManagerBasedRlEnv) -> dict[str, bool]:
    manager = raw_env.termination_manager
    return {
        "success": bool(manager.get_term("success_egg_inside_bucket")[0].item()),
        "fall": bool(manager.get_term("egg_fell")[0].item()),
        "oob": bool(manager.get_term("egg_oob")[0].item()),
        "nan": bool(manager.get_term("nan_state")[0].item()),
        "timeout": bool(manager.get_term("time_out")[0].item()),
    }


def _outcome(flags: dict[str, bool]) -> tuple[str, bool]:
    names = [name for name in ("success", "fall", "oob", "nan", "timeout") if flags[name]]
    if not names:
        return "unknown", False
    if len(names) == 1:
        return names[0], False
    return "overlap:" + "+".join(names), True


def _run_episode(
    vec_env: RslRlVecEnvWrapper,
    policy,
    cfg: Stage1EvaluateConfig,
    condition: Condition,
    identity: dict[str, object],
    episode_index: int,
    video_file: str,
) -> dict[str, object]:
    seed_rng(cfg.evaluation_seed, torch_deterministic=True)
    obs, initial = _reset_to_condition(
        vec_env, condition, cfg.evaluation_seed
    )
    if hasattr(policy, "reset"):
        policy.reset()

    initial_xy = float(
        np.linalg.norm(initial["egg"][:2] - initial["bucket"][:2])
    )
    initial_3d = float(np.linalg.norm(initial["egg"] - initial["bucket"]))
    min_xy = initial_xy
    min_3d = initial_3d
    max_egg_z = float(initial["egg"][2])
    episode_return = 0.0
    trajectory_hash = hashlib.sha256()
    final = initial
    flags = {name: False for name in ("success", "fall", "oob", "nan", "timeout")}

    raw_env = vec_env.unwrapped
    max_steps = raw_env.max_episode_length + 1
    for _ in range(max_steps):
        with torch.no_grad():
            if cfg.policy_mode == "deterministic":
                actions = policy(obs)
            else:
                actions = policy(obs, stochastic_output=True)

        obs, reward, dones, _ = vec_env.step(actions)
        episode_return += float(reward[0].item())
        final = _snapshot(raw_env)

        xy_distance = float(
            np.linalg.norm(final["egg"][:2] - final["bucket"][:2])
        )
        distance_3d = float(np.linalg.norm(final["egg"] - final["bucket"]))
        min_xy = _finite_min(min_xy, xy_distance)
        min_3d = _finite_min(min_3d, distance_3d)
        max_egg_z = _finite_max(max_egg_z, float(final["egg"][2]))

        fingerprint_values = np.concatenate(
            (
                actions[0].detach().cpu().numpy().astype(np.float64),
                np.asarray([float(reward[0].item())]),
                final["egg"],
                final["tendon_len"],
                final["tendon_command"],
            )
        )
        trajectory_hash.update(np.round(fingerprint_values, 7).tobytes())

        if bool(dones[0].item()):
            flags = _termination_flags(raw_env)
            break
    else:
        raise RuntimeError(
            f"Episode exceeded {max_steps} steps without a registered termination."
        )

    outcome, overlap = _outcome(flags)
    steps = int(raw_env.episode_length_buf[0].item())
    now = datetime.now(timezone.utc).isoformat()

    row: dict[str, object] = {
        **identity,
        "evaluated_at_utc": now,
        "episode_index": episode_index,
        "protocol_phase": condition.phase,
        "phase_episode_index": condition.phase_episode_index,
        "physical_condition_id": condition.physical_condition_id,
        "egg_offset_x_mm": condition.offset_x_mm,
        "egg_offset_y_mm": condition.offset_y_mm,
        "initial_egg_x_m": initial["egg"][0],
        "initial_egg_y_m": initial["egg"][1],
        "initial_egg_z_m": initial["egg"][2],
        "initial_bucket_site_x_m": initial["bucket"][0],
        "initial_bucket_site_y_m": initial["bucket"][1],
        "initial_bucket_site_z_m": initial["bucket"][2],
        "initial_base_x_m": initial["base"][0],
        "initial_base_y_m": initial["base"][1],
        "initial_base_z_m": initial["base"][2],
        "initial_tendon_0_length_m": initial["tendon_len"][0],
        "initial_tendon_1_length_m": initial["tendon_len"][1],
        "initial_tendon_0_velocity_m_s": initial["tendon_vel"][0],
        "initial_tendon_1_velocity_m_s": initial["tendon_vel"][1],
        "initial_tendon_0_command_m": initial["tendon_command"][0],
        "initial_tendon_1_command_m": initial["tendon_command"][1],
        "outcome": outcome,
        **flags,
        "termination_overlap": overlap,
        "episode_steps": steps,
        "episode_length_s": steps * raw_env.step_dt,
        "return": episode_return,
        "min_egg_bucket_xy_distance_m": min_xy,
        "min_egg_bucket_3d_distance_m": min_3d,
        "max_egg_z_m": max_egg_z,
        "final_egg_x_m": final["egg"][0],
        "final_egg_y_m": final["egg"][1],
        "final_egg_z_m": final["egg"][2],
        "final_bucket_site_x_m": final["bucket"][0],
        "final_bucket_site_y_m": final["bucket"][1],
        "final_bucket_site_z_m": final["bucket"][2],
        "final_base_x_m": final["base"][0],
        "final_base_y_m": final["base"][1],
        "final_base_z_m": final["base"][2],
        "final_tendon_0_length_m": final["tendon_len"][0],
        "final_tendon_1_length_m": final["tendon_len"][1],
        "final_tendon_0_command_m": final["tendon_command"][0],
        "final_tendon_1_command_m": final["tendon_command"][1],
        "trajectory_fingerprint_sha256": trajectory_hash.hexdigest(),
        "video_file": video_file,
    }
    return row


def _termination_flag_arrays(
    raw_env: ManagerBasedRlEnv,
) -> dict[str, np.ndarray]:
    manager = raw_env.termination_manager
    return {
        "success": (
            manager.get_term("success_egg_inside_bucket")
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        ),
        "fall": (
            manager.get_term("egg_fell").detach().cpu().numpy().astype(bool)
        ),
        "oob": (
            manager.get_term("egg_oob").detach().cpu().numpy().astype(bool)
        ),
        "nan": (
            manager.get_term("nan_state").detach().cpu().numpy().astype(bool)
        ),
        "timeout": (
            manager.get_term("time_out").detach().cpu().numpy().astype(bool)
        ),
    }


def _run_stage2_batch(
    vec_env: RslRlVecEnvWrapper,
    policy,
    cfg: Stage1EvaluateConfig,
    conditions: Sequence[Stage2Condition],
    identity: dict[str, object],
    episode_index_start: int,
) -> list[dict[str, object]]:
    """Run one parallel S2-A batch and retain each first terminal state."""
    if cfg.evaluation_seed is None:
        raise RuntimeError("S2-A evaluation seed was not resolved")
    seed_rng(cfg.evaluation_seed, torch_deterministic=True)
    obs, initial_all, actual_egg_offsets, actual_pedestal_offsets = (
        _reset_s2_batch(vec_env, conditions, cfg.evaluation_seed)
    )
    if hasattr(policy, "reset"):
        policy.reset()

    active_count = len(conditions)
    initial = {
        name: values[:active_count].copy()
        for name, values in initial_all.items()
    }
    final = {name: values.copy() for name, values in initial.items()}
    initial_xy = np.linalg.norm(
        initial["egg"][:, :2] - initial["bucket"][:, :2], axis=1
    )
    initial_3d = np.linalg.norm(initial["egg"] - initial["bucket"], axis=1)
    min_xy = initial_xy.copy()
    min_3d = initial_3d.copy()
    max_egg_z = initial["egg"][:, 2].copy()
    episode_return = np.zeros(active_count, dtype=np.float64)
    trajectory_hashes = [hashlib.sha256() for _ in range(active_count)]
    done = np.zeros(active_count, dtype=bool)
    episode_steps = np.zeros(active_count, dtype=np.int64)
    flags_by_episode = [
        {name: False for name in ("success", "fall", "oob", "nan", "timeout")}
        for _ in range(active_count)
    ]

    raw_env = vec_env.unwrapped
    max_steps = raw_env.max_episode_length + 1
    for _ in range(max_steps):
        unfinished_before_step = ~done
        with torch.no_grad():
            if cfg.policy_mode == "deterministic":
                actions = policy(obs)
            else:
                actions = policy(obs, stochastic_output=True)

        obs, reward, dones, _ = vec_env.step(actions)
        snapshot = _snapshot_batch(raw_env)
        reward_np = reward.detach().cpu().numpy().reshape(-1).astype(np.float64)
        actions_np = actions.detach().cpu().numpy().astype(np.float64)

        active_indices = np.flatnonzero(unfinished_before_step)
        episode_return[active_indices] += reward_np[active_indices]
        xy_distance = np.linalg.norm(
            snapshot["egg"][:active_count, :2]
            - snapshot["bucket"][:active_count, :2],
            axis=1,
        )
        distance_3d = np.linalg.norm(
            snapshot["egg"][:active_count]
            - snapshot["bucket"][:active_count],
            axis=1,
        )
        for local_index in active_indices:
            if np.isfinite(xy_distance[local_index]):
                min_xy[local_index] = min(
                    min_xy[local_index], xy_distance[local_index]
                )
            if np.isfinite(distance_3d[local_index]):
                min_3d[local_index] = min(
                    min_3d[local_index], distance_3d[local_index]
                )
            egg_z = snapshot["egg"][local_index, 2]
            if np.isfinite(egg_z):
                max_egg_z[local_index] = max(max_egg_z[local_index], egg_z)
            fingerprint_values = np.concatenate(
                (
                    actions_np[local_index],
                    np.asarray([reward_np[local_index]]),
                    snapshot["egg"][local_index],
                    snapshot["tendon_len"][local_index],
                    snapshot["tendon_command"][local_index],
                )
            )
            trajectory_hashes[local_index].update(
                np.round(fingerprint_values, 7).tobytes()
            )

        dones_np = dones.detach().cpu().numpy().reshape(-1).astype(bool)
        newly_done = unfinished_before_step & dones_np[:active_count]
        if np.any(newly_done):
            flag_arrays = _termination_flag_arrays(raw_env)
            lengths = (
                raw_env.episode_length_buf.detach().cpu().numpy().astype(np.int64)
            )
            for local_index in np.flatnonzero(newly_done):
                flags_by_episode[local_index] = {
                    name: bool(values[local_index])
                    for name, values in flag_arrays.items()
                }
                episode_steps[local_index] = lengths[local_index]
                for name in final:
                    final[name][local_index] = snapshot[name][local_index]
        done |= newly_done
        if np.all(done):
            break

        # auto_reset=False preserves the true terminal state for recording, but
        # MJLab requires every done environment to be reset before the next
        # step. Reset all done simulator slots, including padding slots and
        # already-recorded slots; only unfinished active rows remain scientific
        # evaluation episodes.
        reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            reset_obs_dict, _ = raw_env.reset(env_ids=reset_ids)
            obs = TensorDict(reset_obs_dict, batch_size=[raw_env.num_envs])
    else:
        incomplete = np.flatnonzero(~done).tolist()
        raise RuntimeError(
            f"S2-A batch exceeded {max_steps} steps; incomplete rows {incomplete}"
        )

    rows: list[dict[str, object]] = []
    for local_index, condition in enumerate(conditions):
        flags = flags_by_episode[local_index]
        outcome, overlap = _outcome(flags)
        pair_error_mm = 1000.0 * float(
            np.max(
                np.abs(
                    actual_egg_offsets[local_index]
                    - actual_pedestal_offsets[local_index]
                )
            )
        )
        row: dict[str, object] = {
            **identity,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "episode_index": episode_index_start + local_index,
            "protocol_phase": condition.phase,
            "phase_episode_index": condition.phase_episode_index,
            "physical_condition_id": condition.physical_condition_id,
            "spatial_stratum": condition.spatial_stratum,
            "stratum_x_index": condition.stratum_x_index,
            "stratum_y_index": condition.stratum_y_index,
            "condition_repeat_index": condition.repeat_index,
            "is_unique_continuous_spawn": condition.is_unique_continuous_spawn,
            "egg_offset_x_mm": condition.offset_x_mm,
            "egg_offset_y_mm": condition.offset_y_mm,
            "actual_egg_offset_x_mm": (
                actual_egg_offsets[local_index, 0] * 1000.0
            ),
            "actual_egg_offset_y_mm": (
                actual_egg_offsets[local_index, 1] * 1000.0
            ),
            "actual_pedestal_offset_x_mm": (
                actual_pedestal_offsets[local_index, 0] * 1000.0
            ),
            "actual_pedestal_offset_y_mm": (
                actual_pedestal_offsets[local_index, 1] * 1000.0
            ),
            "egg_pedestal_offset_pair_error_mm": pair_error_mm,
            "initial_egg_x_m": initial["egg"][local_index, 0],
            "initial_egg_y_m": initial["egg"][local_index, 1],
            "initial_egg_z_m": initial["egg"][local_index, 2],
            "initial_bucket_site_x_m": initial["bucket"][local_index, 0],
            "initial_bucket_site_y_m": initial["bucket"][local_index, 1],
            "initial_bucket_site_z_m": initial["bucket"][local_index, 2],
            "initial_base_x_m": initial["base"][local_index, 0],
            "initial_base_y_m": initial["base"][local_index, 1],
            "initial_base_z_m": initial["base"][local_index, 2],
            "initial_tendon_0_length_m": initial["tendon_len"][local_index, 0],
            "initial_tendon_1_length_m": initial["tendon_len"][local_index, 1],
            "initial_tendon_0_velocity_m_s": initial["tendon_vel"][local_index, 0],
            "initial_tendon_1_velocity_m_s": initial["tendon_vel"][local_index, 1],
            "initial_tendon_0_command_m": initial["tendon_command"][local_index, 0],
            "initial_tendon_1_command_m": initial["tendon_command"][local_index, 1],
            "outcome": outcome,
            **flags,
            "termination_overlap": overlap,
            "episode_steps": int(episode_steps[local_index]),
            "episode_length_s": (
                int(episode_steps[local_index]) * raw_env.step_dt
            ),
            "return": episode_return[local_index],
            "min_egg_bucket_xy_distance_m": min_xy[local_index],
            "min_egg_bucket_3d_distance_m": min_3d[local_index],
            "max_egg_z_m": max_egg_z[local_index],
            "final_egg_x_m": final["egg"][local_index, 0],
            "final_egg_y_m": final["egg"][local_index, 1],
            "final_egg_z_m": final["egg"][local_index, 2],
            "final_bucket_site_x_m": final["bucket"][local_index, 0],
            "final_bucket_site_y_m": final["bucket"][local_index, 1],
            "final_bucket_site_z_m": final["bucket"][local_index, 2],
            "final_base_x_m": final["base"][local_index, 0],
            "final_base_y_m": final["base"][local_index, 1],
            "final_base_z_m": final["base"][local_index, 2],
            "final_tendon_0_length_m": final["tendon_len"][local_index, 0],
            "final_tendon_1_length_m": final["tendon_len"][local_index, 1],
            "final_tendon_0_command_m": final["tendon_command"][local_index, 0],
            "final_tendon_1_command_m": final["tendon_command"][local_index, 1],
            "trajectory_fingerprint_sha256": (
                trajectory_hashes[local_index].hexdigest()
            ),
            "video_file": "",
        }
        rows.append(row)
    return rows


def _truth(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _make_basin_figure(rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    grid_rows = [row for row in rows if row["protocol_phase"] == "grid"]
    if not grid_rows:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "success": "#2e7d32",
        "fall": "#d84315",
        "timeout": "#757575",
        "oob": "#6a1b9a",
        "nan": "#111111",
        "overlap": "#c2185b",
        "unknown": "#f9a825",
    }
    labels = {
        "success": "S",
        "fall": "F",
        "timeout": "T",
        "oob": "O",
        "nan": "N",
        "overlap": "X",
        "unknown": "?",
    }

    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    for row in grid_rows:
        raw_outcome = str(row["outcome"])
        category = "overlap" if raw_outcome.startswith("overlap:") else raw_outcome
        x = int(row["egg_offset_x_mm"])
        y = int(row["egg_offset_y_mm"])
        ax.scatter(
            x,
            y,
            s=920,
            marker="s",
            color=colors.get(category, colors["unknown"]),
            edgecolor="white",
            linewidth=1.5,
            zorder=2,
        )
        ax.text(
            x,
            y,
            labels.get(category, "?"),
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
            zorder=3,
        )

    success_points = sum(_truth(row["success"]) for row in grid_rows)
    first = grid_rows[0]
    ax.set_title(
        f"Stage-1 deterministic 25-point outcome map\n"
        f"{first['run_id']} | {first['checkpoint_name']} | "
        f"success at {success_points}/25 exact positions"
    )
    ax.set_xlabel("Egg initial x offset from nominal (mm)")
    ax.set_ylabel("Egg initial y offset from nominal (mm)")
    ax.set_xticks(S1_GRID_OFFSETS_MM)
    ax.set_yticks(S1_GRID_OFFSETS_MM)
    ax.set_xlim(-13, 13)
    ax.set_ylim(-13, 13)
    ax.set_aspect("equal")
    ax.grid(True, color="#d0d0d0", linewidth=0.8, zorder=0)
    ax.scatter(
        0,
        0,
        s=1120,
        marker="s",
        facecolors="none",
        edgecolors="#1565c0",
        linewidth=2.5,
        zorder=4,
        label="Nominal start",
    )
    ax.legend(loc="upper right", frameon=False)
    fig.text(
        0.5,
        0.015,
        "One deterministic rollout per exact grid point. "
        "Markers are observed outcomes, not estimated probabilities.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    png = output_dir / "success_basin.png"
    svg = output_dir / "success_basin.svg"
    fig.savefig(png, dpi=200)
    fig.savefig(svg)
    plt.close(fig)
    return [png, svg]


def _make_summary(rows: list[dict[str, object]], video_files: list[Path]) -> dict[str, object]:
    nominal = [row for row in rows if row["protocol_phase"] == "nominal_repeat"]
    grid = [row for row in rows if row["protocol_phase"] == "grid"]
    expected_grid = {
        (dx, dy) for dy in S1_GRID_OFFSETS_MM for dx in S1_GRID_OFFSETS_MM
    }
    observed_grid = {
        (int(row["egg_offset_x_mm"]), int(row["egg_offset_y_mm"]))
        for row in grid
    }
    oob_count = sum(_truth(row["oob"]) for row in rows)
    nan_count = sum(_truth(row["nan"]) for row in rows)
    overlap_count = sum(_truth(row["termination_overlap"]) for row in rows)
    nominal_all_success = len(nominal) == 5 and all(
        _truth(row["success"]) for row in nominal
    )
    grid_complete = len(grid) == 25 and observed_grid == expected_grid
    video_present = bool(video_files)

    return {
        "protocol_version": S1_PROTOCOL_VERSION,
        "row_count": len(rows),
        "nominal_repeat_rows": len(nominal),
        "nominal_physical_condition_count": len(
            {str(row["physical_condition_id"]) for row in nominal}
        ),
        "nominal_success_rows": sum(_truth(row["success"]) for row in nominal),
        "nominal_unique_trajectory_fingerprints": len(
            {str(row["trajectory_fingerprint_sha256"]) for row in nominal}
        ),
        "nominal_all_success": nominal_all_success,
        "grid_rows": len(grid),
        "grid_complete": grid_complete,
        "grid_success_points": sum(_truth(row["success"]) for row in grid),
        "grid_fall_points": sum(_truth(row["fall"]) for row in grid),
        "grid_timeout_points": sum(_truth(row["timeout"]) for row in grid),
        "grid_oob_points": sum(_truth(row["oob"]) for row in grid),
        "grid_nan_points": sum(_truth(row["nan"]) for row in grid),
        "oob_terminations_all_rows": oob_count,
        "nan_terminations_all_rows": nan_count,
        "termination_overlap_rows": overlap_count,
        "nominal_video_present": video_present,
        "nominal_video_files": [path.name for path in video_files],
        "checkpoint_s1_gate_pass": (
            nominal_all_success
            and grid_complete
            and oob_count == 0
            and nan_count == 0
            and overlap_count == 0
            and video_present
        ),
        "interpretation_note": (
            "The five nominal rows test computational repeatability at one "
            "physical condition. The 25 grid rows are 25 distinct exact "
            "conditions; grid_success_points is coverage, not a success probability."
        ),
    }


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """95% Wilson interval for a Bernoulli proportion."""
    if total <= 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return (
        float(max(0.0, centre - half_width)),
        float(min(1.0, centre + half_width)),
    )


def _stratified_interval(
    stratum_successes: Sequence[int],
    stratum_totals: Sequence[int],
) -> tuple[float, float]:
    """Approximate 95% CI for an equal-area stratified success estimate."""
    if not stratum_totals or any(total < 2 for total in stratum_totals):
        return (float("nan"), float("nan"))
    weights = np.full(len(stratum_totals), 1.0 / len(stratum_totals))
    proportions = np.asarray(stratum_successes, dtype=np.float64) / np.asarray(
        stratum_totals, dtype=np.float64
    )
    sample_variances = (
        proportions
        * (1.0 - proportions)
        * np.asarray(stratum_totals, dtype=np.float64)
        / (np.asarray(stratum_totals, dtype=np.float64) - 1.0)
    )
    estimate = float(np.sum(weights * proportions))
    variance = float(
        np.sum(
            weights
            * weights
            * sample_variances
            / np.asarray(stratum_totals, dtype=np.float64)
        )
    )
    half_width = 1.959963984540054 * np.sqrt(max(0.0, variance))
    return (
        float(max(0.0, estimate - half_width)),
        float(min(1.0, estimate + half_width)),
    )


def _write_dict_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_s2_heatmap(
    values: np.ndarray,
    *,
    title: str,
    note: str,
    ticks: Sequence[float],
    output_stem: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    tick_step = float(ticks[1] - ticks[0])
    lower_edge = float(ticks[0]) - 0.5 * tick_step
    upper_edge = float(ticks[-1]) + 0.5 * tick_step
    image = ax.imshow(
        values,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
        extent=(lower_edge, upper_edge, lower_edge, upper_edge),
        interpolation="nearest",
        aspect="equal",
    )
    for y_index, y_value in enumerate(ticks):
        for x_index, x_value in enumerate(ticks):
            value = values[y_index, x_index]
            ax.text(
                x_value,
                y_value,
                f"{100.0 * value:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )
    ax.set_title(title)
    ax.set_xlabel("Egg and pedestal x offset from nominal (mm)")
    ax.set_ylabel("Egg and pedestal y offset from nominal (mm)")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    fig.colorbar(image, ax=ax, label="Observed success fraction")
    fig.text(0.5, 0.015, note, ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    png = output_stem.with_suffix(".png")
    svg = output_stem.with_suffix(".svg")
    fig.savefig(png, dpi=200)
    fig.savefig(svg)
    plt.close(fig)
    return [png, svg]


def _make_s2_summary(
    rows: list[dict[str, object]],
    protocol: Literal["s2a-smoke", "s2a"],
    output_dir: Path,
) -> tuple[dict[str, object], list[Path]]:
    nominal = [row for row in rows if row["protocol_phase"] == "nominal_repeat"]
    continuous = [
        row for row in rows if row["protocol_phase"] == "continuous_stratified"
    ]
    exact = [
        row for row in rows if row["protocol_phase"] == "exact_grid_repeat"
    ]
    if protocol == "s2a":
        expected_nominal = S2A_NOMINAL_REPEATS
        expected_continuous_per_stratum = S2A_CONTINUOUS_SAMPLES_PER_STRATUM
        expected_grid_repeats = S2A_GRID_REPEATS_PER_POINT
    else:
        expected_nominal = S2A_SMOKE_NOMINAL_REPEATS
        expected_continuous_per_stratum = (
            S2A_SMOKE_CONTINUOUS_SAMPLES_PER_STRATUM
        )
        expected_grid_repeats = S2A_SMOKE_GRID_REPEATS_PER_POINT

    stratum_rows: list[dict[str, object]] = []
    stratum_successes: list[int] = []
    stratum_totals: list[int] = []
    stratum_matrix = np.full(
        (S2A_STRATA_PER_AXIS, S2A_STRATA_PER_AXIS), np.nan
    )
    cell_width_mm = 2.0 * S2A_RANGE_MM / S2A_STRATA_PER_AXIS
    for stratum in range(S2A_STRATA_PER_AXIS * S2A_STRATA_PER_AXIS):
        selected = [
            row for row in continuous if int(row["spatial_stratum"]) == stratum
        ]
        successes = sum(_truth(row["success"]) for row in selected)
        total = len(selected)
        rate = successes / total if total else float("nan")
        lower, upper = _wilson_interval(successes, total)
        stratum_x = stratum % S2A_STRATA_PER_AXIS
        stratum_y = stratum // S2A_STRATA_PER_AXIS
        stratum_matrix[stratum_y, stratum_x] = rate
        stratum_successes.append(successes)
        stratum_totals.append(total)
        stratum_rows.append(
            {
                "spatial_stratum": stratum,
                "stratum_x_index": stratum_x,
                "stratum_y_index": stratum_y,
                "x_min_mm": -S2A_RANGE_MM + stratum_x * cell_width_mm,
                "x_max_mm": -S2A_RANGE_MM + (stratum_x + 1) * cell_width_mm,
                "y_min_mm": -S2A_RANGE_MM + stratum_y * cell_width_mm,
                "y_max_mm": -S2A_RANGE_MM + (stratum_y + 1) * cell_width_mm,
                "episodes": total,
                "successes": successes,
                "falls": sum(_truth(row["fall"]) for row in selected),
                "timeouts": sum(_truth(row["timeout"]) for row in selected),
                "oob": sum(_truth(row["oob"]) for row in selected),
                "nan": sum(_truth(row["nan"]) for row in selected),
                "success_rate": rate,
                "wilson_95_lower": lower,
                "wilson_95_upper": upper,
                "meets_80_percent_gate": (
                    rate >= S2A_STRATUM_SUCCESS_RATE_MIN if total else False
                ),
            }
        )
    strata_csv = output_dir / "continuous_strata_summary.csv"
    _write_dict_csv(strata_csv, stratum_rows)

    grid_rows: list[dict[str, object]] = []
    grid_matrix = np.full((5, 5), np.nan)
    for y_index, offset_y_mm in enumerate(S2A_GRID_OFFSETS_MM):
        for x_index, offset_x_mm in enumerate(S2A_GRID_OFFSETS_MM):
            if offset_x_mm == 0.0 and offset_y_mm == 0.0:
                selected = nominal
                source_phase = "nominal_repeat"
            else:
                selected = [
                    row
                    for row in exact
                    if float(row["egg_offset_x_mm"]) == offset_x_mm
                    and float(row["egg_offset_y_mm"]) == offset_y_mm
                ]
                source_phase = "exact_grid_repeat"
            successes = sum(_truth(row["success"]) for row in selected)
            total = len(selected)
            fraction = successes / total if total else float("nan")
            grid_matrix[y_index, x_index] = fraction
            if protocol == "s2a":
                if fraction >= 0.90:
                    classification = "supported"
                elif fraction < 0.10:
                    classification = "unsupported"
                else:
                    classification = "borderline"
            else:
                classification = "diagnostic_only"
            grid_rows.append(
                {
                    "offset_x_mm": offset_x_mm,
                    "offset_y_mm": offset_y_mm,
                    "source_phase": source_phase,
                    "computational_repeats": total,
                    "successes": successes,
                    "falls": sum(_truth(row["fall"]) for row in selected),
                    "timeouts": sum(_truth(row["timeout"]) for row in selected),
                    "oob": sum(_truth(row["oob"]) for row in selected),
                    "nan": sum(_truth(row["nan"]) for row in selected),
                    "computational_success_fraction": fraction,
                    "classification": classification,
                }
            )
    grid_csv = output_dir / "exact_grid_repeatability_summary.csv"
    _write_dict_csv(grid_csv, grid_rows)

    continuous_successes = sum(_truth(row["success"]) for row in continuous)
    continuous_rate = (
        continuous_successes / len(continuous) if continuous else float("nan")
    )
    stratified_lower, stratified_upper = _stratified_interval(
        stratum_successes, stratum_totals
    )
    nominal_successes = sum(_truth(row["success"]) for row in nominal)
    unique_continuous = len(
        {
            (float(row["egg_offset_x_mm"]), float(row["egg_offset_y_mm"]))
            for row in continuous
        }
    )
    expected_continuous = (
        S2A_STRATA_PER_AXIS
        * S2A_STRATA_PER_AXIS
        * expected_continuous_per_stratum
    )
    expected_exact = (
        (len(S2A_GRID_OFFSETS_MM) ** 2 - 1) * expected_grid_repeats
    )
    oob_count = sum(_truth(row["oob"]) for row in rows)
    nan_count = sum(_truth(row["nan"]) for row in rows)
    overlap_count = sum(_truth(row["termination_overlap"]) for row in rows)
    pair_error_max = max(
        (float(row["egg_pedestal_offset_pair_error_mm"]) for row in rows),
        default=float("nan"),
    )
    full_protocol_complete = (
        len(nominal) == expected_nominal
        and len(continuous) == expected_continuous
        and unique_continuous == expected_continuous
        and all(total == expected_continuous_per_stratum for total in stratum_totals)
        and len(exact) == expected_exact
        and all(
            int(row["computational_repeats"])
            == (
                expected_nominal
                if float(row["offset_x_mm"]) == 0.0
                and float(row["offset_y_mm"]) == 0.0
                else expected_grid_repeats
            )
            for row in grid_rows
        )
    )
    full_gate_applicable = protocol == "s2a" and full_protocol_complete
    gate_pass: bool | None = None
    if full_gate_applicable:
        gate_pass = (
            nominal_successes >= S2A_NOMINAL_SUCCESS_MIN
            and continuous_rate >= S2A_CONTINUOUS_SUCCESS_RATE_MIN
            and all(
                total == S2A_CONTINUOUS_SAMPLES_PER_STRATUM
                and successes / total >= S2A_STRATUM_SUCCESS_RATE_MIN
                for successes, total in zip(
                    stratum_successes, stratum_totals, strict=True
                )
            )
            and oob_count == 0
            and nan_count == 0
            and overlap_count == 0
        )

    stratum_centres = tuple(
        -S2A_RANGE_MM + (index + 0.5) * cell_width_mm
        for index in range(S2A_STRATA_PER_AXIS)
    )
    figure_paths = [
        *_make_s2_heatmap(
            stratum_matrix,
            title="S2-A continuous-spawn success by spatial stratum",
            note=(
                "Each cell uses distinct uniform position samples; this map "
                "estimates spatial-distribution performance."
            ),
            ticks=stratum_centres,
            output_stem=output_dir / "continuous_strata_success",
        ),
        *_make_s2_heatmap(
            grid_matrix,
            title="S2-A exact-grid computational repeatability",
            note=(
                "Repeated deterministic rollouts at one exact condition are "
                "computational-repeatability checks, not independent physical trials."
            ),
            ticks=S2A_GRID_OFFSETS_MM,
            output_stem=output_dir / "exact_grid_repeatability",
        ),
    ]

    summary = {
        "protocol_version": S2A_PROTOCOL_VERSION,
        "protocol": protocol,
        "row_count": len(rows),
        "nominal_repeat_rows": len(nominal),
        "nominal_success_rows": nominal_successes,
        "nominal_unique_trajectory_fingerprints": len(
            {str(row["trajectory_fingerprint_sha256"]) for row in nominal}
        ),
        "continuous_rows": len(continuous),
        "continuous_unique_positions": unique_continuous,
        "continuous_successes": continuous_successes,
        "continuous_stratified_success_estimate": continuous_rate,
        "continuous_stratified_95_lower": stratified_lower,
        "continuous_stratified_95_upper": stratified_upper,
        "continuous_minimum_stratum_success_rate": min(
            (successes / total for successes, total in zip(
                stratum_successes, stratum_totals, strict=True
            ) if total),
            default=float("nan"),
        ),
        "exact_grid_noncentre_rows": len(exact),
        "exact_grid_points_including_nominal_centre": len(grid_rows),
        "outcomes_by_phase": {
            phase_name: {
                outcome_name: sum(_truth(row[outcome_name]) for row in phase_rows)
                for outcome_name in ("success", "fall", "timeout", "oob", "nan")
            }
            for phase_name, phase_rows in (
                ("nominal_repeat", nominal),
                ("continuous_stratified", continuous),
                ("exact_grid_repeat", exact),
            )
        },
        "oob_terminations_all_rows": oob_count,
        "nan_terminations_all_rows": nan_count,
        "termination_overlap_rows": overlap_count,
        "maximum_egg_pedestal_offset_pair_error_mm": pair_error_max,
        "protocol_complete": full_protocol_complete,
        "s2a_gate_applicable": full_gate_applicable,
        "checkpoint_s2a_gate_pass": gate_pass,
        "gate_definition": {
            "nominal_success_minimum": "90/100",
            "continuous_overall_success_minimum": "90%",
            "continuous_each_stratum_success_minimum": "80%",
            "oob": 0,
            "nan": 0,
            "termination_overlap": 0,
        },
        "interpretation_note": (
            "The continuous stratified estimate refers to the defined uniform "
            "position distribution over the +/-2 mm square. Nominal and exact-grid "
            "fractions describe computational repeatability under MuJoCo Warp; "
            "they are not physical-world success probabilities."
        ),
    }
    return summary, [strata_csv, grid_csv, *figure_paths]


def _write_manifest(output_dir: Path) -> Path:
    manifest = output_dir / "manifest.sha256"
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path != manifest
    )
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in files
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def run_evaluation(cfg: Stage1EvaluateConfig) -> Path:
    is_s2 = cfg.protocol in {"s2a-smoke", "s2a"}
    expected_seed = S2A_EVALUATION_SEED if is_s2 else S1_EVALUATION_SEED
    if cfg.evaluation_seed is None:
        cfg = replace(cfg, evaluation_seed=expected_seed)
    assert cfg.evaluation_seed is not None

    if not re.fullmatch(r"[0-9a-fA-F]{40}", cfg.training_git_commit):
        raise ValueError(
            "--training-git-commit must be the full 40-character commit SHA."
        )
    if cfg.protocol in {"s1", "s2a-smoke", "s2a"} and (
        cfg.policy_mode != "deterministic"
    ):
        raise ValueError(
            f"The {cfg.protocol} protocol requires --policy-mode deterministic."
        )
    if cfg.protocol in {"s1", "s2a-smoke", "s2a"} and (
        cfg.evaluation_seed != expected_seed
    ):
        raise ValueError(
            f"{cfg.protocol} is preregistered with evaluation seed {expected_seed}."
        )
    if cfg.protocol == "s1" and not cfg.record_video:
        raise ValueError("The S1 protocol requires --record-video.")
    if cfg.record_video and cfg.protocol == "grid":
        raise ValueError(
            "A required nominal video cannot be produced by grid-only evaluation."
        )
    if is_s2 and cfg.record_video:
        raise ValueError(
            "S2-A batch evaluation does not record video; use the separate "
            "visual reset check for visual evidence."
        )
    if is_s2 and cfg.s2_num_envs < 1:
        raise ValueError("--s2-num-envs must be at least 1.")

    repo_root = _repo_root()
    evaluator_commit = _run_git("rev-parse", "HEAD")
    dirty_output = _run_git("status", "--porcelain")
    evaluator_dirty = bool(dirty_output)
    if evaluator_dirty and not cfg.allow_dirty_evaluator:
        raise RuntimeError(
            "Evaluator checkout is dirty. Commit the evaluator and run from a "
            "clean checkout, or explicitly use --allow-dirty-evaluator for a "
            "diagnostic run.\n" + dirty_output
        )

    output_dir = Path(cfg.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use a new checkpoint-specific directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_torch_backends(allow_tf32=True, deterministic=True)
    seed_rng(cfg.evaluation_seed, torch_deterministic=True)
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    task_id = STAGE2_TASK_ID if is_s2 else STAGE1_TASK_ID
    protocol_version = S2A_PROTOCOL_VERSION if is_s2 else S1_PROTOCOL_VERSION
    env_cfg = load_env_cfg(task_id, play=False)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = cfg.s2_num_envs if is_s2 else 1
    env_cfg.auto_reset = False
    env_cfg.seed = cfg.evaluation_seed
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.observations["critic"].enable_corruption = False
    if is_s2:
        removed_event = env_cfg.events.pop("stage2_egg_pedestal_spawn", None)
        if removed_event is None:
            raise RuntimeError(
                "Stage-2 spawn event was not found; evaluator cannot impose "
                "controlled conditions safely."
            )

    checkpoint = _resolve_checkpoint(cfg, agent_cfg.experiment_name)
    checkpoint_hash = _sha256_file(checkpoint)
    checkpoint_size = checkpoint.stat().st_size
    task_hash = _sha256_tree(repo_root / "src" / "spirob_mjlab")
    lock_path = repo_root / "uv.lock"
    lock_hash = _sha256_file(lock_path) if lock_path.is_file() else "missing"

    video_dir = output_dir / "videos"
    video_prefix = f"{_slug(cfg.run_id)}-{_slug(checkpoint.stem)}-nominal"
    expected_video = video_dir / f"{video_prefix}-episode-0.mp4"

    render_mode = "rgb_array" if cfg.record_video else None
    base_env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
        render_mode=render_mode,
    )
    wrapped_env = base_env
    if cfg.record_video:
        wrapped_env = VideoRecorder(
            base_env,
            video_folder=video_dir,
            episode_trigger=lambda episode: episode == 0,
            video_length=None,
            name_prefix=video_prefix,
        )
    vec_env = RslRlVecEnvWrapper(wrapped_env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    identity: dict[str, object] = {
        "protocol_version": protocol_version,
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
        "checkpoint_size_bytes": checkpoint_size,
        "policy_mode": cfg.policy_mode,
    }

    csv_path = output_dir / "episodes.csv"
    rows: list[dict[str, object]] = []
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            if is_s2:
                s2_conditions = _stage2_conditions(cfg.protocol, cfg.evaluation_seed)
                total_batches = (
                    len(s2_conditions) + cfg.s2_num_envs - 1
                ) // cfg.s2_num_envs
                for batch_number, start in enumerate(
                    range(0, len(s2_conditions), cfg.s2_num_envs), start=1
                ):
                    batch_conditions = s2_conditions[start : start + cfg.s2_num_envs]
                    batch_rows = _run_stage2_batch(
                        vec_env,
                        policy,
                        cfg,
                        batch_conditions,
                        identity,
                        start,
                    )
                    writer.writerows(batch_rows)
                    stream.flush()
                    rows.extend(batch_rows)
                    batch_successes = sum(
                        _truth(row["success"]) for row in batch_rows
                    )
                    print(
                        f"[S2-A] batch {batch_number}/{total_batches}: "
                        f"episodes {start + 1}-{start + len(batch_rows)}, "
                        f"success {batch_successes}/{len(batch_rows)}"
                    )
            else:
                s1_conditions = _conditions(cfg.protocol)
                for index, condition in enumerate(s1_conditions):
                    video_file = (
                        expected_video.relative_to(output_dir).as_posix()
                        if cfg.record_video
                        and condition.phase == "nominal_repeat"
                        and condition.phase_episode_index == 0
                        else ""
                    )
                    row = _run_episode(
                        vec_env,
                        policy,
                        cfg,
                        condition,
                        identity,
                        index,
                        video_file,
                    )
                    writer.writerow(row)
                    stream.flush()
                    rows.append(row)
                    print(
                        f"[S1] {index + 1:02d}/{len(s1_conditions):02d} "
                        f"{condition.phase} {condition.physical_condition_id}: "
                        f"{row['outcome']}"
                    )
    finally:
        vec_env.close()

    video_files = sorted(video_dir.glob("*.mp4")) if video_dir.exists() else []
    if is_s2:
        summary, supplemental_paths = _make_s2_summary(
            rows, cfg.protocol, output_dir
        )
    else:
        supplemental_paths = _make_basin_figure(rows, output_dir)
        summary = _make_summary(rows, video_files)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    checkpoint_hash_path = output_dir / "checkpoint.sha256"
    checkpoint_hash_path.write_text(
        f"{checkpoint_hash}  {checkpoint.name}\n", encoding="utf-8"
    )

    metadata = {
        "protocol_version": protocol_version,
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
        "checkpoint_size_bytes": checkpoint_size,
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
            "MJLab 1.4.0 notes that MuJoCo Warp is not fully deterministic. "
            "Repeated nominal and exact-grid rollouts therefore test computational "
            "repeatability; they are not independent physical conditions. The "
            "continuous stratified rows use distinct sampled physical conditions."
        ),
        "s2_batch_execution_note": (
            "With auto_reset disabled, each terminal simulator slot is recorded "
            "before an explicit partial reset. Finished and padding slots may run "
            "additional ignored episodes while unfinished scientific rows continue."
            if is_s2
            else None
        ),
        "generated_files": [
            csv_path.name,
            summary_path.name,
            checkpoint_hash_path.name,
            *[
                path.relative_to(output_dir).as_posix()
                for path in supplemental_paths
            ],
            *[path.relative_to(output_dir).as_posix() for path in video_files],
        ],
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _write_manifest(output_dir)

    label = "S2-A" if is_s2 else "S1"
    gate_key = "checkpoint_s2a_gate_pass" if is_s2 else "checkpoint_s1_gate_pass"
    print(f"[{label}] Episode CSV: {csv_path}")
    print(f"[{label}] Summary: {summary_path}")
    print(f"[{label}] Checkpoint SHA-256: {checkpoint_hash}")
    print(f"[{label}] Checkpoint gate pass: {summary[gate_key]}")
    return output_dir


def main() -> None:
    cfg = tyro.cli(Stage1EvaluateConfig)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
