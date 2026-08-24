"""Episode-level evaluator for the frozen SpiRob Stage-1 policy.

This tool implements the S1 protocol without changing the training task:

1. five deterministic nominal repeats (one physical initial condition), then
2. one deterministic rollout at each point of the fixed 5 x 5 egg-offset grid.

It writes one CSV row per completed episode, a checkpoint hash, provenance
metadata, one nominal video when requested, and categorical success-basin
figures.  The grid is evaluation-only; it is not training randomization.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

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


TASK_ID = "Mjlab-SpiRob-EggToBucket-Stage1"
PROTOCOL_VERSION = "S1-v1"
S1_EVALUATION_SEED = 1001
S1_GRID_OFFSETS_MM = (-10, -5, 0, 5, 10)


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
    # Initial condition (environment-local coordinates).
    "egg_offset_x_mm",
    "egg_offset_y_mm",
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
    """Configuration for one frozen-checkpoint S1 evaluation."""

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

    evaluation_seed: int = S1_EVALUATION_SEED
    """Fixed evaluator seed. Use the same value for all three training seeds."""

    protocol: Literal["nominal", "grid", "s1"] = "s1"
    """S1 runs nominal first and then the grid."""

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
        "protocol_version": PROTOCOL_VERSION,
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
    if not re.fullmatch(r"[0-9a-fA-F]{40}", cfg.training_git_commit):
        raise ValueError(
            "--training-git-commit must be the full 40-character commit SHA."
        )
    if cfg.protocol == "s1" and cfg.policy_mode != "deterministic":
        raise ValueError("The S1 protocol requires --policy-mode deterministic.")
    if cfg.protocol == "s1" and cfg.evaluation_seed != S1_EVALUATION_SEED:
        raise ValueError(
            f"{PROTOCOL_VERSION} is pre-registered with evaluation seed "
            f"{S1_EVALUATION_SEED}."
        )
    if cfg.protocol == "s1" and not cfg.record_video:
        raise ValueError("The S1 protocol requires --record-video.")
    if cfg.record_video and cfg.protocol == "grid":
        raise ValueError(
            "A required nominal video cannot be produced by grid-only evaluation."
        )

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

    env_cfg = load_env_cfg(TASK_ID, play=False)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = 1
    env_cfg.auto_reset = False
    env_cfg.seed = cfg.evaluation_seed
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.observations["critic"].enable_corruption = False

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

    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

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
        "checkpoint_size_bytes": checkpoint_size,
        "policy_mode": cfg.policy_mode,
    }

    csv_path = output_dir / "episodes.csv"
    rows: list[dict[str, object]] = []
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for index, condition in enumerate(_conditions(cfg.protocol)):
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
                    f"[S1] {index + 1:02d}/{len(_conditions(cfg.protocol)):02d} "
                    f"{condition.phase} {condition.physical_condition_id}: "
                    f"{row['outcome']}"
                )
    finally:
        vec_env.close()

    figure_paths = _make_basin_figure(rows, output_dir)
    video_files = sorted(video_dir.glob("*.mp4")) if video_dir.exists() else []
    summary = _make_summary(rows, video_files)
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
            "The five nominal repeats therefore test computational repeatability; "
            "they are not five independent physical conditions."
        ),
        "generated_files": [
            csv_path.name,
            summary_path.name,
            checkpoint_hash_path.name,
            *[path.name for path in figure_paths],
            *[path.relative_to(output_dir).as_posix() for path in video_files],
        ],
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _write_manifest(output_dir)

    print(f"[S1] Episode CSV: {csv_path}")
    print(f"[S1] Summary: {summary_path}")
    print(f"[S1] Checkpoint SHA-256: {checkpoint_hash}")
    print(f"[S1] Checkpoint gate pass: {summary['checkpoint_s1_gate_pass']}")
    return output_dir


def main() -> None:
    cfg = tyro.cli(Stage1EvaluateConfig)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
