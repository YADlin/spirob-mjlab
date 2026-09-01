"""Diagnose rare numerical divergence in the frozen S2-C controller.

This protocol is deliberately diagnostic.  It does not train, alter the frozen
checkpoint, certify a workspace, or replace S2-C-workspace-map-v1.

Full mode first replays the two original 256-environment batch contexts that
contained the NaN episodes in the archived workspace map.  It then concentrates
2,000 repeats at the coordinate where V1 observed a fresh NaN, with smaller
replications at the second implicated coordinate and adjacent/mirrored controls.
Whenever a NaN termination occurs, the evaluator saves a 16-step scalar trace
and the last fully finite simulator arrays so that the numerical precursor can
be inspected instead of merely counted. V2 converts native MuJoCo-Warp arrays
through Warp's supported PyTorch bridge and refuses to run if the required
generalized-state instrumentation is unavailable.
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
from typing import Literal, Sequence

import numpy as np
import torch
import tyro
import warp as wp
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.random import seed_rng
from mjlab.utils.torch import configure_torch_backends

import evaluate_stage1 as stage2_base
import evaluate_stage2c_workspace as workspace_base
import mjlab.tasks  # noqa: F401  # Populate MJLab's built-in registry.
import spirob_mjlab  # noqa: F401  # Register the SpiRob tasks.


PROTOCOL_VERSION = "S2-C-numerical-diagnostic-v2"
TASK_ID = "Mjlab-SpiRob-EggToBucket-Stage2C"
EVALUATION_SEED = 2710
REQUIRED_NUM_ENVS = 256
TRACE_STEPS = 16
SMOKE_REPEATS_PER_POSITION = 5
FULL_REPEATS_BY_LABEL = {
    # V1 observed one new NaN in 500 repeats at this coordinate. Concentrating
    # V2 here gives the corrected raw-state tracer a substantially better
    # chance of capturing another event without repeating the full workspace.
    "observed_p10_y0": 2_000,
    "observed_p10_y6": 500,
    "adjacent_p8_y0": 250,
    "adjacent_p8_y6": 250,
    "mirrored_m10_y0": 250,
    "mirrored_m10_y6": 250,
}
SOURCE_WORKSPACE_ARCHIVE_SHA256 = (
    "0f0b9fee0712161e48e51e71502ac9e5f906530fc3fa89bdd30f38fd9e3e9461"
)
SOURCE_NAN_EPISODE_INDICES = (2404, 8596)

# Two observed coordinates, adjacent x controls, then mirrored x controls.
TARGETED_POSITIONS = (
    ("observed_p10_y0", 10.0, 0.0, "observed_nan_position"),
    ("observed_p10_y6", 10.0, 6.0, "observed_nan_position"),
    ("adjacent_p8_y0", 8.0, 0.0, "adjacent_control"),
    ("adjacent_p8_y6", 8.0, 6.0, "adjacent_control"),
    ("mirrored_m10_y0", -10.0, 0.0, "mirrored_control"),
    ("mirrored_m10_y6", -10.0, 6.0, "mirrored_control"),
)

RAW_STATE_NAMES = (
    "qpos",
    "qvel",
    "qacc",
    "qacc_warmstart",
    "qacc_smooth",
    "ctrl",
    "actuator_force",
    "qfrc_actuator",
    "qfrc_smooth",
    "qfrc_constraint",
    "qfrc_passive",
    "qfrc_bias",
)

REQUIRED_RAW_STATE_NAMES = (
    "qpos",
    "qvel",
    "qacc",
    "qacc_smooth",
    "qfrc_actuator",
    "qfrc_smooth",
    "qfrc_constraint",
)

RAW_SCALAR_NAMES = (
    "solver_niter",
    "nefc",
    "nisland",
    "ncdof",
    "overflow",
)

TRACE_COLUMNS = (
    "relative_step",
    "episode_step",
    "reward",
    "action_0",
    "action_1",
    "egg_x_m",
    "egg_y_m",
    "egg_z_m",
    "tendon_0_length_m",
    "tendon_1_length_m",
    "tendon_0_velocity_m_s",
    "tendon_1_velocity_m_s",
    "tendon_0_command_m",
    "tendon_1_command_m",
    "tendon_0_desired_m",
    "tendon_1_desired_m",
    *tuple(f"max_abs_{name}" for name in RAW_STATE_NAMES),
    *RAW_SCALAR_NAMES,
    "all_recorded_values_finite",
)

EPISODE_COLUMNS = (
    "protocol_version",
    "evaluated_at_utc",
    "run_id",
    "diagnostic_phase",
    "diagnostic_label",
    "control_role",
    "episode_index",
    "source_workspace_episode_index",
    "condition_repeat_index",
    "egg_offset_x_mm",
    "egg_offset_y_mm",
    "outcome_raw",
    "outcome_adjudicated",
    "success",
    "fall",
    "timeout",
    "oob",
    "nan",
    "termination_overlap_raw",
    "success_at_horizon",
    "incompatible_physical_overlap",
    "engineering_invalid",
    "episode_steps",
    "episode_length_s",
    "return",
    "last_finite_step",
    "first_nonfinite_step",
    "first_nonfinite_fields",
    "initial_egg_x_m",
    "initial_egg_y_m",
    "initial_egg_z_m",
    "final_egg_x_m",
    "final_egg_y_m",
    "final_egg_z_m",
    "initial_tendon_0_length_m",
    "initial_tendon_1_length_m",
    "final_tendon_0_length_m",
    "final_tendon_1_length_m",
    "final_tendon_0_command_m",
    "final_tendon_1_command_m",
    "final_tendon_0_desired_m",
    "final_tendon_1_desired_m",
    "trace_csv",
    "state_npz",
)

Mode = Literal["smoke", "full"]


@dataclass(frozen=True)
class NumericalDiagnosticConfig:
    """Inputs for the frozen S2-C numerical-stability diagnostic."""

    run_id: str
    training_seed: int
    training_git_commit: str
    checkpoint_file: str
    expected_checkpoint_sha256: str
    source_workspace_archive_sha256: str
    output_dir: str
    mode: Mode = "smoke"
    evaluation_seed: int = EVALUATION_SEED
    device: str | None = None
    num_envs: int = REQUIRED_NUM_ENVS
    allow_dirty_evaluator: bool = False


@dataclass(frozen=True)
class DiagnosticCondition:
    condition: stage2_base.Stage2Condition
    diagnostic_phase: str
    diagnostic_label: str
    control_role: str
    source_workspace_episode_index: int | None = None


def _targeted_conditions(mode: Mode) -> list[DiagnosticCondition]:
    conditions: list[DiagnosticCondition] = []
    for label, offset_x_mm, offset_y_mm, control_role in TARGETED_POSITIONS:
        repeats = (
            FULL_REPEATS_BY_LABEL[label]
            if mode == "full"
            else SMOKE_REPEATS_PER_POSITION
        )
        for repeat_index in range(repeats):
            conditions.append(
                DiagnosticCondition(
                    condition=stage2_base.Stage2Condition(
                        phase="exact_grid_repeat",
                        phase_episode_index=-1,
                        offset_x_mm=offset_x_mm,
                        offset_y_mm=offset_y_mm,
                        spatial_stratum=-1,
                        stratum_x_index=-1,
                        stratum_y_index=-1,
                        repeat_index=repeat_index,
                        is_unique_continuous_spawn=False,
                    ),
                    diagnostic_phase="targeted_repeat",
                    diagnostic_label=label,
                    control_role=control_role,
                )
            )
    rng = np.random.default_rng(EVALUATION_SEED + 1)
    rng.shuffle(conditions)
    return conditions


def _original_batch_contexts() -> list[DiagnosticCondition]:
    original = workspace_base._workspace_conditions("full")
    starts = sorted(
        {
            index // REQUIRED_NUM_ENVS * REQUIRED_NUM_ENVS
            for index in SOURCE_NAN_EPISODE_INDICES
        }
    )
    contexts: list[DiagnosticCondition] = []
    for start in starts:
        batch_number = start // REQUIRED_NUM_ENVS + 1
        for source_index in range(start, start + REQUIRED_NUM_ENVS):
            condition = original[source_index]
            label = f"original_workspace_batch_{batch_number:02d}"
            role = (
                "source_nan_episode"
                if source_index in SOURCE_NAN_EPISODE_INDICES
                else "original_batch_context"
            )
            contexts.append(
                DiagnosticCondition(
                    condition=condition,
                    diagnostic_phase="original_batch_replay",
                    diagnostic_label=label,
                    control_role=role,
                    source_workspace_episode_index=source_index,
                )
            )
    return contexts


def _diagnostic_conditions(mode: Mode) -> list[DiagnosticCondition]:
    specs = []
    if mode == "full":
        specs.extend(_original_batch_contexts())
    specs.extend(_targeted_conditions(mode))
    return [
        replace(
            spec,
            condition=replace(spec.condition, phase_episode_index=index),
        )
        for index, spec in enumerate(specs)
    ]


def _available_raw_state(
    raw_env: ManagerBasedRlEnv,
    active_count: int,
) -> dict[str, torch.Tensor]:
    """Expose MJWarp arrays as Torch tensors without silently dropping them."""
    available: dict[str, torch.Tensor] = {}
    for name in RAW_STATE_NAMES:
        value = getattr(raw_env.sim.data, name, None)
        if value is None:
            continue
        source = getattr(value, "wp_array", value)
        tensor = value if torch.is_tensor(value) else wp.to_torch(source)
        if tensor.ndim >= 2 and tensor.shape[0] >= active_count:
            available[name] = tensor[:active_count]
    return available


def _available_raw_scalars(
    raw_env: ManagerBasedRlEnv,
    active_count: int,
) -> dict[str, torch.Tensor]:
    available: dict[str, torch.Tensor] = {}
    for name in RAW_SCALAR_NAMES:
        value = getattr(raw_env.sim.data, name, None)
        if value is None:
            continue
        source = getattr(value, "wp_array", value)
        tensor = value if torch.is_tensor(value) else wp.to_torch(source)
        if tensor.ndim == 1 and tensor.shape[0] >= active_count:
            available[name] = tensor[:active_count]
    return available


def _require_raw_instrumentation(
    raw_state: dict[str, torch.Tensor],
    raw_scalars: dict[str, torch.Tensor],
) -> None:
    missing = [name for name in REQUIRED_RAW_STATE_NAMES if name not in raw_state]
    if missing:
        raise RuntimeError(
            "MJWarp raw-state instrumentation is incomplete; missing: "
            + ", ".join(missing)
        )
    if "solver_niter" not in raw_scalars or "nefc" not in raw_scalars:
        raise RuntimeError(
            "MJWarp solver instrumentation requires solver_niter and nefc"
        )


def _tendon_desired_targets(
    raw_env: ManagerBasedRlEnv,
    active_count: int,
) -> np.ndarray:
    action_term = raw_env.action_manager.get_term("cable_len")
    desired = getattr(action_term, "desired_action", None)
    if not torch.is_tensor(desired):
        raise RuntimeError("cable_len action term does not expose desired_action")
    return (
        desired[:active_count]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def _max_abs_rows(value: torch.Tensor) -> np.ndarray:
    flat = value.detach().reshape(value.shape[0], -1)
    return torch.amax(torch.abs(flat), dim=1).cpu().numpy().astype(np.float64)


def _snapshot_finite(snapshot: dict[str, np.ndarray], active_count: int) -> np.ndarray:
    finite = np.ones(active_count, dtype=bool)
    for value in snapshot.values():
        selected = np.asarray(value[:active_count])
        finite &= np.isfinite(selected.reshape(active_count, -1)).all(axis=1)
    return finite


def _adjudicate_outcome(flags: dict[str, bool]) -> tuple[str, bool, bool, bool]:
    """Return outcome, success-at-horizon, physical-overlap, engineering-invalid."""
    physical = [name for name in ("success", "fall", "oob", "nan") if flags[name]]
    timeout = flags["timeout"]
    success_at_horizon = physical == ["success"] and timeout
    incompatible_physical_overlap = len(physical) > 1

    if success_at_horizon:
        outcome = "success_at_horizon"
    elif len(physical) == 1:
        outcome = physical[0] + ("_at_horizon" if timeout else "")
    elif incompatible_physical_overlap:
        outcome = "incompatible:" + "+".join(physical)
        if timeout:
            outcome += "+timeout"
    elif timeout:
        outcome = "timeout"
    else:
        outcome = "unknown"

    engineering_invalid = (
        flags["nan"]
        or flags["oob"]
        or incompatible_physical_overlap
        or outcome == "unknown"
    )
    return (
        outcome,
        success_at_horizon,
        incompatible_physical_overlap,
        engineering_invalid,
    )


def _chronological_trace(
    ring: np.ndarray,
    ring_next: int,
    ring_count: int,
) -> np.ndarray:
    if ring_count == 0:
        return ring[:0]
    start = (ring_next - ring_count) % TRACE_STEPS
    indices = [(start + offset) % TRACE_STEPS for offset in range(ring_count)]
    return ring[indices].copy()


def _write_trace_csv(path: Path, trace: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRACE_COLUMNS)
        writer.writeheader()
        for relative_step, values in enumerate(trace, start=1 - len(trace)):
            row = {name: values[index] for index, name in enumerate(TRACE_COLUMNS)}
            row["relative_step"] = relative_step
            writer.writerow(row)


def _trace_vector(
    step: int,
    local_index: int,
    reward: np.ndarray,
    actions: np.ndarray,
    snapshot: dict[str, np.ndarray],
    raw_maxima: dict[str, np.ndarray],
    raw_scalars: dict[str, np.ndarray],
    all_finite: bool,
) -> np.ndarray:
    def action_component(index: int) -> float:
        if actions.shape[1] > index:
            return float(actions[local_index, index])
        return float("nan")

    return np.asarray(
        [
            0.0,
            float(step),
            float(reward[local_index]),
            action_component(0),
            action_component(1),
            *snapshot["egg"][local_index].tolist(),
            *snapshot["tendon_len"][local_index].tolist(),
            *snapshot["tendon_vel"][local_index].tolist(),
            *snapshot["tendon_command"][local_index].tolist(),
            *snapshot["tendon_desired"][local_index].tolist(),
            *[
                float(raw_maxima.get(name, np.full(len(reward), np.nan))[local_index])
                for name in RAW_STATE_NAMES
            ],
            *[
                float(raw_scalars.get(name, np.full(len(reward), np.nan))[local_index])
                for name in RAW_SCALAR_NAMES
            ],
            float(all_finite),
        ],
        dtype=np.float64,
    )


def _run_diagnostic_batch(
    vec_env: RslRlVecEnvWrapper,
    policy,
    cfg: NumericalDiagnosticConfig,
    specs: Sequence[DiagnosticCondition],
    episode_index_start: int,
    output_dir: Path,
) -> tuple[list[dict[str, object]], list[Path]]:
    seed_rng(cfg.evaluation_seed, torch_deterministic=True)
    base_conditions = [spec.condition for spec in specs]
    obs, initial_all, _, _ = stage2_base._reset_s2_batch(
        vec_env, base_conditions, cfg.evaluation_seed
    )
    if hasattr(policy, "reset"):
        policy.reset()

    active_count = len(specs)
    initial = {
        name: values[:active_count].copy() for name, values in initial_all.items()
    }
    initial["tendon_desired"] = _tendon_desired_targets(
        vec_env.unwrapped, active_count
    )
    final = {name: values.copy() for name, values in initial.items()}
    last_finite_snapshot = {name: values.copy() for name, values in initial.items()}
    done = np.zeros(active_count, dtype=bool)
    episode_steps = np.zeros(active_count, dtype=np.int64)
    episode_return = np.zeros(active_count, dtype=np.float64)
    last_finite_step = np.zeros(active_count, dtype=np.int64)
    first_nonfinite_step = np.full(active_count, -1, dtype=np.int64)
    first_nonfinite_fields: list[list[str]] = [[] for _ in range(active_count)]
    flags_by_episode = [
        {name: False for name in ("success", "fall", "oob", "nan", "timeout")}
        for _ in range(active_count)
    ]

    ring = np.full((active_count, TRACE_STEPS, len(TRACE_COLUMNS)), np.nan)
    ring_next = np.zeros(active_count, dtype=np.int64)
    ring_count = np.zeros(active_count, dtype=np.int64)

    raw_initial = _available_raw_state(vec_env.unwrapped, active_count)
    raw_scalar_initial = _available_raw_scalars(vec_env.unwrapped, active_count)
    _require_raw_instrumentation(raw_initial, raw_scalar_initial)
    last_finite_raw = {
        name: value.detach().clone() for name, value in raw_initial.items()
    }
    last_finite_raw_scalars = {
        name: value.detach().clone() for name, value in raw_scalar_initial.items()
    }
    nan_artifacts: dict[int, tuple[str, str]] = {}
    generated: list[Path] = []

    raw_env = vec_env.unwrapped
    max_steps = raw_env.max_episode_length + 1
    for control_step in range(1, max_steps + 1):
        unfinished_before_step = ~done
        with torch.no_grad():
            actions = policy(obs)

        obs, reward, dones, _ = vec_env.step(actions)
        snapshot = stage2_base._snapshot_batch(raw_env)
        snapshot["tendon_desired"] = _tendon_desired_targets(
            raw_env, active_count
        )
        reward_np = reward.detach().cpu().numpy().reshape(-1).astype(np.float64)
        actions_np = actions.detach().cpu().numpy().astype(np.float64)
        active_indices = np.flatnonzero(unfinished_before_step)
        episode_return[active_indices] += reward_np[active_indices]

        raw_state = _available_raw_state(raw_env, active_count)
        raw_scalar_state = _available_raw_scalars(raw_env, active_count)
        raw_maxima = {name: _max_abs_rows(value) for name, value in raw_state.items()}
        raw_scalar_values = {
            name: value.detach().cpu().numpy().astype(np.float64)
            for name, value in raw_scalar_state.items()
        }
        raw_finite = np.ones(active_count, dtype=bool)
        for value in raw_state.values():
            finite = (
                torch.isfinite(value.detach().reshape(active_count, -1))
                .all(dim=1)
                .cpu()
                .numpy()
            )
            raw_finite &= finite
        snapshot_finite = _snapshot_finite(snapshot, active_count)
        action_finite = np.isfinite(actions_np[:active_count]).all(axis=1)
        reward_finite = np.isfinite(reward_np[:active_count])
        all_finite = raw_finite & snapshot_finite & action_finite & reward_finite

        for local_index in active_indices:
            trace_values = _trace_vector(
                control_step,
                local_index,
                reward_np,
                actions_np,
                snapshot,
                raw_maxima,
                raw_scalar_values,
                bool(all_finite[local_index]),
            )
            slot = int(ring_next[local_index])
            ring[local_index, slot] = trace_values
            ring_next[local_index] = (slot + 1) % TRACE_STEPS
            ring_count[local_index] = min(
                int(ring_count[local_index]) + 1, TRACE_STEPS
            )

            if all_finite[local_index]:
                last_finite_step[local_index] = control_step
                for name in last_finite_snapshot:
                    last_finite_snapshot[name][local_index] = snapshot[name][
                        local_index
                    ]
                for name, value in raw_state.items():
                    last_finite_raw[name][local_index] = value[local_index].detach()
                for name, value in raw_scalar_state.items():
                    last_finite_raw_scalars[name][local_index] = (
                        value[local_index].detach()
                    )
            elif first_nonfinite_step[local_index] < 0:
                first_nonfinite_step[local_index] = control_step
                fields: list[str] = []
                for name, value in raw_state.items():
                    if not bool(torch.isfinite(value[local_index]).all().item()):
                        fields.append(name)
                for name, value in snapshot.items():
                    if not np.isfinite(value[local_index]).all():
                        fields.append(name)
                if not action_finite[local_index]:
                    fields.append("action")
                if not reward_finite[local_index]:
                    fields.append("reward")
                first_nonfinite_fields[local_index] = fields

        dones_np = dones.detach().cpu().numpy().reshape(-1).astype(bool)
        newly_done = unfinished_before_step & dones_np[:active_count]
        if np.any(newly_done):
            flag_arrays = stage2_base._termination_flag_arrays(raw_env)
            lengths = (
                raw_env.episode_length_buf.detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            for local_index in np.flatnonzero(newly_done):
                flags = {
                    name: bool(values[local_index])
                    for name, values in flag_arrays.items()
                }
                flags_by_episode[local_index] = flags
                episode_steps[local_index] = lengths[local_index]
                for name in final:
                    final[name][local_index] = snapshot[name][local_index]

                if flags["nan"]:
                    episode_index = episode_index_start + local_index
                    stem = f"nan_episode_{episode_index:05d}"
                    trace_path = output_dir / "nan_traces" / f"{stem}_trace.csv"
                    state_path = output_dir / "nan_traces" / f"{stem}_state.npz"
                    trace = _chronological_trace(
                        ring[local_index],
                        int(ring_next[local_index]),
                        int(ring_count[local_index]),
                    )
                    _write_trace_csv(trace_path, trace)
                    state_payload: dict[str, np.ndarray] = {
                        "last_finite_step": np.asarray(
                            last_finite_step[local_index]
                        ),
                        "first_nonfinite_step": np.asarray(
                            first_nonfinite_step[local_index]
                        ),
                        "triggering_action": actions_np[local_index],
                    }
                    for name, value in last_finite_snapshot.items():
                        state_payload[f"last_finite_{name}"] = value[local_index]
                        state_payload[f"terminal_{name}"] = snapshot[name][local_index]
                    for name, value in last_finite_raw.items():
                        state_payload[f"last_finite_{name}"] = (
                            value[local_index].detach().cpu().numpy()
                        )
                    for name, value in raw_state.items():
                        state_payload[f"terminal_{name}"] = (
                            value[local_index].detach().cpu().numpy()
                        )
                    for name, value in last_finite_raw_scalars.items():
                        state_payload[f"last_finite_{name}"] = (
                            value[local_index].detach().cpu().numpy()
                        )
                    for name, value in raw_scalar_state.items():
                        state_payload[f"terminal_{name}"] = (
                            value[local_index].detach().cpu().numpy()
                        )
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(state_path, **state_payload)
                    generated.extend((trace_path, state_path))
                    nan_artifacts[local_index] = (
                        trace_path.relative_to(output_dir).as_posix(),
                        state_path.relative_to(output_dir).as_posix(),
                    )
        done |= newly_done
        if np.all(done):
            break

        reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            reset_obs_dict, _ = raw_env.reset(env_ids=reset_ids)
            obs = TensorDict(reset_obs_dict, batch_size=[raw_env.num_envs])
    else:
        incomplete = np.flatnonzero(~done).tolist()
        raise RuntimeError(
            "Numerical diagnostic exceeded the episode horizon; "
            f"incomplete rows {incomplete}"
        )

    rows: list[dict[str, object]] = []
    now = datetime.now(timezone.utc).isoformat()
    for local_index, spec in enumerate(specs):
        flags = flags_by_episode[local_index]
        raw_outcome, raw_overlap = stage2_base._outcome(flags)
        (
            adjudicated,
            success_at_horizon,
            incompatible_physical_overlap,
            engineering_invalid,
        ) = _adjudicate_outcome(flags)
        trace_csv, state_npz = nan_artifacts.get(local_index, ("", ""))
        row: dict[str, object] = {
            "protocol_version": PROTOCOL_VERSION,
            "evaluated_at_utc": now,
            "run_id": cfg.run_id,
            "diagnostic_phase": spec.diagnostic_phase,
            "diagnostic_label": spec.diagnostic_label,
            "control_role": spec.control_role,
            "episode_index": episode_index_start + local_index,
            "source_workspace_episode_index": (
                ""
                if spec.source_workspace_episode_index is None
                else spec.source_workspace_episode_index
            ),
            "condition_repeat_index": spec.condition.repeat_index,
            "egg_offset_x_mm": spec.condition.offset_x_mm,
            "egg_offset_y_mm": spec.condition.offset_y_mm,
            "outcome_raw": raw_outcome,
            "outcome_adjudicated": adjudicated,
            **flags,
            "termination_overlap_raw": raw_overlap,
            "success_at_horizon": success_at_horizon,
            "incompatible_physical_overlap": incompatible_physical_overlap,
            "engineering_invalid": engineering_invalid,
            "episode_steps": int(episode_steps[local_index]),
            "episode_length_s": int(episode_steps[local_index]) * raw_env.step_dt,
            "return": episode_return[local_index],
            "last_finite_step": int(last_finite_step[local_index]),
            "first_nonfinite_step": (
                ""
                if first_nonfinite_step[local_index] < 0
                else int(first_nonfinite_step[local_index])
            ),
            "first_nonfinite_fields": "+".join(
                first_nonfinite_fields[local_index]
            ),
            "initial_egg_x_m": initial["egg"][local_index, 0],
            "initial_egg_y_m": initial["egg"][local_index, 1],
            "initial_egg_z_m": initial["egg"][local_index, 2],
            "final_egg_x_m": final["egg"][local_index, 0],
            "final_egg_y_m": final["egg"][local_index, 1],
            "final_egg_z_m": final["egg"][local_index, 2],
            "initial_tendon_0_length_m": initial["tendon_len"][local_index, 0],
            "initial_tendon_1_length_m": initial["tendon_len"][local_index, 1],
            "final_tendon_0_length_m": final["tendon_len"][local_index, 0],
            "final_tendon_1_length_m": final["tendon_len"][local_index, 1],
            "final_tendon_0_command_m": final["tendon_command"][local_index, 0],
            "final_tendon_1_command_m": final["tendon_command"][local_index, 1],
            "final_tendon_0_desired_m": final["tendon_desired"][local_index, 0],
            "final_tendon_1_desired_m": final["tendon_desired"][local_index, 1],
            "trace_csv": trace_csv,
            "state_npz": state_npz,
        }
        rows.append(row)
    return rows, generated


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _summarize(rows: list[dict[str, object]], expected_rows: int) -> dict[str, object]:
    labels = sorted({str(row["diagnostic_label"]) for row in rows})
    by_condition: dict[str, object] = {}
    for label in labels:
        selected = [row for row in rows if row["diagnostic_label"] == label]
        nan_count = sum(_truth(row["nan"]) for row in selected)
        lower, upper = stage2_base._wilson_interval(nan_count, len(selected))
        by_condition[label] = {
            "rows": len(selected),
            "offset_x_mm": sorted({float(row["egg_offset_x_mm"]) for row in selected}),
            "offset_y_mm": sorted({float(row["egg_offset_y_mm"]) for row in selected}),
            "nan_count": nan_count,
            "nan_rate": nan_count / len(selected) if selected else None,
            "nan_rate_wilson_95_lower": lower,
            "nan_rate_wilson_95_upper": upper,
            "successes": sum(_truth(row["success"]) for row in selected),
            "success_at_horizon": sum(
                _truth(row["success_at_horizon"]) for row in selected
            ),
            "engineering_invalid": sum(
                _truth(row["engineering_invalid"]) for row in selected
            ),
        }

    nan_count = sum(_truth(row["nan"]) for row in rows)
    trace_count = sum(bool(row["trace_csv"]) and bool(row["state_npz"]) for row in rows)
    protocol_complete = len(rows) == expected_rows
    return {
        "protocol_version": PROTOCOL_VERSION,
        "row_count": len(rows),
        "expected_rows": expected_rows,
        "protocol_complete": protocol_complete,
        "nan_count": nan_count,
        "nan_rate": nan_count / len(rows) if rows else None,
        "nan_trace_count": trace_count,
        "all_nan_events_traced": trace_count == nan_count,
        "success_at_horizon_count": sum(
            _truth(row["success_at_horizon"]) for row in rows
        ),
        "incompatible_physical_overlap_count": sum(
            _truth(row["incompatible_physical_overlap"]) for row in rows
        ),
        "engineering_invalid_count": sum(
            _truth(row["engineering_invalid"]) for row in rows
        ),
        "diagnostic_complete": protocol_complete and trace_count == nan_count,
        "raw_state_instrumentation_complete": True,
        "raw_state_fields": list(RAW_STATE_NAMES),
        "raw_scalar_fields": list(RAW_SCALAR_NAMES),
        "requires_numerical_follow_up": nan_count > 0,
        "by_condition": by_condition,
        "outcome_rule": {
            "success_plus_timeout": "success_at_horizon; raw flags retained",
            "single_physical_outcome_plus_timeout": "physical outcome at horizon",
            "multiple_physical_outcomes": "engineering-invalid overlap",
            "nan": "engineering-invalid",
            "oob": "engineering-invalid",
        },
        "interpretation_note": (
            "This is a post-map numerical cause-finding experiment. Targeted "
            "rates are selection-conditioned and must not be presented as "
            "unbiased workspace-wide probabilities or as workspace certification."
        ),
        "trace_resolution_note": (
            "Raw-state traces are sampled at policy/control-step boundaries. "
            "They can expose the last finite magnitudes and first observed "
            "non-finite fields, but not the exact physics substep within a "
            "control interval."
        ),
    }


def run_diagnostic(cfg: NumericalDiagnosticConfig) -> Path:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", cfg.training_git_commit):
        raise ValueError("--training-git-commit must be a full 40-character SHA")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", cfg.expected_checkpoint_sha256):
        raise ValueError("--expected-checkpoint-sha256 must contain 64 hex characters")
    if cfg.source_workspace_archive_sha256.lower() != SOURCE_WORKSPACE_ARCHIVE_SHA256:
        raise ValueError(
            "Source workspace archive hash does not match the frozen full_v1 evidence"
        )
    if cfg.evaluation_seed != EVALUATION_SEED:
        raise ValueError(
            f"Numerical diagnostic requires evaluation seed {EVALUATION_SEED}"
        )
    if cfg.num_envs != REQUIRED_NUM_ENVS:
        raise ValueError(
            f"Numerical diagnostic requires exactly {REQUIRED_NUM_ENVS} environments"
        )

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
    lock_hash = (
        stage2_base._sha256_file(lock_path)
        if lock_path.is_file()
        else "missing"
    )

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

    specs = _diagnostic_conditions(cfg.mode)
    rows: list[dict[str, object]] = []
    generated: list[Path] = []
    csv_path = output_dir / "episodes.csv"
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=EPISODE_COLUMNS)
            writer.writeheader()
            total_batches = (len(specs) + cfg.num_envs - 1) // cfg.num_envs
            for batch_number, start in enumerate(
                range(0, len(specs), cfg.num_envs), start=1
            ):
                batch_specs = specs[start : start + cfg.num_envs]
                batch_rows, batch_generated = _run_diagnostic_batch(
                    vec_env,
                    policy,
                    cfg,
                    batch_specs,
                    start,
                    output_dir,
                )
                writer.writerows(batch_rows)
                stream.flush()
                rows.extend(batch_rows)
                generated.extend(batch_generated)
                nan_count = sum(_truth(row["nan"]) for row in batch_rows)
                print(
                    f"[S2-C numerical] batch {batch_number}/{total_batches}: "
                    f"NaN {nan_count}/{len(batch_rows)}"
                )
    finally:
        vec_env.close()

    summary = _summarize(rows, len(specs))
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
        "source_workspace_archive_sha256": cfg.source_workspace_archive_sha256,
        "source_nan_episode_indices": list(SOURCE_NAN_EPISODE_INDICES),
        "raw_state_fields": list(RAW_STATE_NAMES),
        "required_raw_state_fields": list(REQUIRED_RAW_STATE_NAMES),
        "raw_scalar_fields": list(RAW_SCALAR_NAMES),
        "targeted_positions": [
            {
                "label": label,
                "offset_x_mm": x,
                "offset_y_mm": y,
                "control_role": role,
            }
            for label, x, y, role in TARGETED_POSITIONS
        ],
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
            *[path.relative_to(output_dir).as_posix() for path in generated],
        ],
        "known_reproducibility_limit": (
            "Mean policy actions are used, but MuJoCo Warp GPU rollouts may not "
            "be bitwise repeatable. Original-batch replay tests reproducibility; "
            "targeted repeats investigate conditional numerical behaviour."
        ),
        "trace_resolution_limit": (
            "The ring buffer samples after each policy/control step, not after "
            "each internal MuJoCo physics substep. A substep-level replay is a "
            "separate follow-up only if V2 captures another event."
        ),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    stage2_base._write_manifest(output_dir)

    print(f"[S2-C numerical] Episodes: {csv_path}")
    print(f"[S2-C numerical] Summary: {summary_path}")
    print(f"[S2-C numerical] Protocol complete: {summary['protocol_complete']}")
    print(f"[S2-C numerical] NaN events: {summary['nan_count']}")
    print(f"[S2-C numerical] All NaN events traced: {summary['all_nan_events_traced']}")
    return output_dir


def main() -> None:
    run_diagnostic(tyro.cli(NumericalDiagnosticConfig))


if __name__ == "__main__":
    main()
