"""MDP terms for the SpiRob egg-to-bucket task.

All terms are vectorized over parallel mjlab environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from spirob_mjlab.sector_curriculum import (
    RETENTION_CELL_LOWER_LEFT_MM,
    RETENTION_CELL_SIZE_MM,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

ROBOT_CFG = SceneEntityCfg(
    "robot",
    tendon_names=("cable_0", "cable_1"),
    site_names=("tip_site",),
)
BUCKET_CFG = SceneEntityCfg("bucket", site_names=("bucket_site",))

_S2_SPAWN_OFFSET_BUFFER = "_stage2_spawn_offset_xy"
_S2_SPAWN_NOMINAL_BUFFER = "_stage2_spawn_is_nominal"
_S2_SPAWN_STRATUM_BUFFER = "_stage2_spawn_stratum"
_S2_SPAWN_REJECTION_BUFFER = "_stage2_spawn_rejection_count"
_S2_SPAWN_GROUP_BUFFER = "_stage2_spawn_group"
_S2_RESET_COUNT_BUFFER = "_stage2_reset_count"
_S2_CORE_RESET_COUNT_BUFFER = "_stage2_core_reset_count"
_S2_EXPANDED_RESET_COUNT_BUFFER = "_stage2_expanded_reset_count"
_S2_RETENTION_RESET_COUNT_BUFFER = "_stage2_retention_reset_count"
_S2_SECTOR_RESET_COUNT_BUFFER = "_stage2_sector_reset_count"
_S2_SPAWN_RADIUS_BUFFER = "_stage2_spawn_radius_m"
_S2_SPAWN_ANGLE_BUFFER = "_stage2_spawn_angle_deg"
_S2_SPAWN_RADIAL_STRATUM_BUFFER = "_stage2_spawn_radial_stratum"
_S2_SPAWN_ANGULAR_STRATUM_BUFFER = "_stage2_spawn_angular_stratum"

_S2_GROUP_NOMINAL = 0
_S2_GROUP_CORE_5MM = 1
_S2_GROUP_EXPANDED_10MM = 2
_S2_GROUP_RETENTION = _S2_GROUP_CORE_5MM
_S2_GROUP_SECTOR = _S2_GROUP_EXPANDED_10MM


def _resolve_env_ids(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def _stage2_buffer(
    env: "ManagerBasedRlEnv",
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    value = getattr(env, name, None)
    expected_shape = (env.num_envs, *shape)
    if value is None or tuple(value.shape) != expected_shape:
        value = torch.zeros(expected_shape, device=env.device, dtype=dtype)
        setattr(env, name, value)
    return value


def _stage2_spawn_is_clear(
    egg_xy_local: torch.Tensor,
    *,
    robot_centerline_x: float,
    min_robot_centerline_clearance: float,
    bucket_xy_local: tuple[float, float],
    min_bucket_center_clearance: float,
) -> torch.Tensor:
    """Conservative reset-time clearance screen for the egg/pedestal pair.

    The robot is represented by a vertical centreline exclusion strip because
    its 21 links can bend along the work direction. The bucket is represented
    by a centre-distance exclusion region. These tests prevent obviously
    intersecting initial states; the extreme-corner runtime smoke test remains
    the final check against compiled MuJoCo collision geometry.
    """
    robot_clear = (
        torch.abs(egg_xy_local[:, 0] - robot_centerline_x)
        >= min_robot_centerline_clearance
    )
    bucket_xy = torch.tensor(
        bucket_xy_local,
        device=egg_xy_local.device,
        dtype=egg_xy_local.dtype,
    )
    bucket_clear = (
        torch.linalg.norm(egg_xy_local - bucket_xy, dim=-1)
        >= min_bucket_center_clearance
    )
    return robot_clear & bucket_clear


def reset_stage2_egg_and_pedestal(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    x_range: tuple[float, float] = (-0.002, 0.002),
    y_range: tuple[float, float] = (-0.002, 0.002),
    strata_per_axis: int = 5,
    nominal_every_n: int = 4,
    max_resample_attempts: int = 32,
    robot_centerline_x: float = 0.0,
    min_robot_centerline_clearance: float = 0.038,
    bucket_xy_local: tuple[float, float] = (-0.05, 0.15),
    min_bucket_center_clearance: float = 0.055,
) -> None:
    """Reset egg and pedestal with one shared, balanced continuous XY offset.

    Each environment cycles through a two-dimensional stratum. Within a
    stratum, the offset is sampled continuously and uniformly. One episode in
    every ``nominal_every_n`` is held exactly at the original S1 condition to
    monitor retention of the demonstrated skill.

    Only collision-clear candidates are written to the simulator. This is an
    admissibility screen, not a claim that every admitted point is reachable.
    Reachability is measured later by repeated frozen-policy evaluation.
    """
    if strata_per_axis < 1:
        raise ValueError("strata_per_axis must be at least 1")
    if nominal_every_n < 2:
        raise ValueError("nominal_every_n must be at least 2")
    if x_range[0] > x_range[1] or y_range[0] > y_range[1]:
        raise ValueError("spawn ranges must be ordered (minimum, maximum)")

    env_ids = _resolve_env_ids(env, env_ids)
    num_resets = len(env_ids)
    if num_resets == 0:
        return

    reset_count = _stage2_buffer(
        env,
        _S2_RESET_COUNT_BUFFER,
        shape=(),
        dtype=torch.long,
    )
    local_count = reset_count[env_ids]
    is_nominal = local_count.remainder(nominal_every_n) == 0

    num_strata = strata_per_axis * strata_per_axis
    cycle_index = torch.div(local_count, nominal_every_n, rounding_mode="floor")
    stratum = (env_ids + cycle_index).remainder(num_strata)
    stratum_x = stratum.remainder(strata_per_axis)
    stratum_y = torch.div(stratum, strata_per_axis, rounding_mode="floor")

    x_width = (x_range[1] - x_range[0]) / strata_per_axis
    y_width = (y_range[1] - y_range[0]) / strata_per_axis
    x_low = x_range[0] + stratum_x.to(torch.float32) * x_width
    y_low = y_range[0] + stratum_y.to(torch.float32) * y_width

    offsets = torch.zeros((num_resets, 2), device=env.device)
    rejection_count = torch.zeros(num_resets, device=env.device, dtype=torch.long)
    pending = ~is_nominal

    egg = env.scene["egg"]
    default_egg_state = egg.data.default_root_state[env_ids].clone()
    nominal_egg_xy = default_egg_state[:, :2]
    nominal_is_clear = _stage2_spawn_is_clear(
        nominal_egg_xy,
        robot_centerline_x=robot_centerline_x,
        min_robot_centerline_clearance=min_robot_centerline_clearance,
        bucket_xy_local=bucket_xy_local,
        min_bucket_center_clearance=min_bucket_center_clearance,
    )
    if not torch.all(nominal_is_clear):
        raise RuntimeError("The nominal S1 spawn fails the S2 clearance screen")

    for _ in range(max_resample_attempts):
        if not torch.any(pending):
            break
        sample_count = int(pending.sum().item())
        draw = torch.rand((sample_count, 2), device=env.device)
        candidate_offsets = torch.empty((sample_count, 2), device=env.device)
        candidate_offsets[:, 0] = x_low[pending] + draw[:, 0] * x_width
        candidate_offsets[:, 1] = y_low[pending] + draw[:, 1] * y_width
        candidate_egg_xy = nominal_egg_xy[pending] + candidate_offsets
        valid = _stage2_spawn_is_clear(
            candidate_egg_xy,
            robot_centerline_x=robot_centerline_x,
            min_robot_centerline_clearance=min_robot_centerline_clearance,
            bucket_xy_local=bucket_xy_local,
            min_bucket_center_clearance=min_bucket_center_clearance,
        )
        pending_indices = pending.nonzero(as_tuple=False).squeeze(-1)
        accepted_indices = pending_indices[valid]
        offsets[accepted_indices] = candidate_offsets[valid]
        rejection_count[pending_indices[~valid]] += 1
        pending[accepted_indices] = False

    if torch.any(pending):
        failed_strata = torch.unique(stratum[pending]).tolist()
        raise RuntimeError(
            "No collision-clear Stage-2 spawn found after "
            f"{max_resample_attempts} attempts in strata {failed_strata}. "
            "Do not expand the training range until the admissible region is mapped."
        )

    egg_state = default_egg_state
    egg_state[:, :3] += env.scene.env_origins[env_ids]
    egg_state[:, :2] += offsets
    egg.write_root_state_to_sim(egg_state, env_ids=env_ids)

    pedestal = env.scene["pedestal"]
    if not pedestal.is_mocap:
        raise RuntimeError("Stage-2 pedestal must be configured as a mocap body")
    default_pedestal_state = pedestal.data.default_root_state[env_ids]
    pedestal_pose = torch.empty((num_resets, 7), device=env.device)
    pedestal_pose[:, :3] = (
        default_pedestal_state[:, :3] + env.scene.env_origins[env_ids]
    )
    pedestal_pose[:, :2] += offsets
    pedestal_pose[:, 3:7] = default_pedestal_state[:, 3:7]
    pedestal.write_mocap_pose_to_sim(pedestal_pose, env_ids=env_ids)

    _stage2_buffer(
        env,
        _S2_SPAWN_OFFSET_BUFFER,
        shape=(2,),
        dtype=torch.float32,
    )[env_ids] = offsets
    _stage2_buffer(
        env,
        _S2_SPAWN_NOMINAL_BUFFER,
        shape=(),
        dtype=torch.bool,
    )[env_ids] = is_nominal
    _stage2_buffer(
        env,
        _S2_SPAWN_STRATUM_BUFFER,
        shape=(),
        dtype=torch.long,
    )[env_ids] = torch.where(is_nominal, -1, stratum)
    _stage2_buffer(
        env,
        _S2_SPAWN_REJECTION_BUFFER,
        shape=(),
        dtype=torch.long,
    )[env_ids] = rejection_count
    reset_count[env_ids] += 1


def reset_stage2c_egg_and_pedestal(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    core_half_range_m: float = 0.005,
    expanded_half_range_m: float = 0.010,
    strata_per_axis: int = 5,
    schedule_length: int = 10,
    nominal_slots: int = 1,
    core_slots: int = 3,
    max_resample_attempts: int = 32,
    robot_centerline_x: float = 0.0,
    min_robot_centerline_clearance: float = 0.038,
    bucket_xy_local: tuple[float, float] = (-0.05, 0.15),
    min_bucket_center_clearance: float = 0.055,
) -> None:
    """Reset S2-C with a 10% nominal, 30% core, 60% expanded mixture.

    The egg and pedestal always receive the same offset. Environment identity
    shifts the deterministic ten-reset schedule so a parallel reset batch is
    already close to the intended mixture. Each individual environment still
    receives the exact mixture over every ten of its resets.
    """
    if strata_per_axis < 1:
        raise ValueError("strata_per_axis must be at least 1")
    if not 0.0 < core_half_range_m < expanded_half_range_m:
        raise ValueError("require 0 < core range < expanded range")
    if schedule_length < 3:
        raise ValueError("schedule_length must be at least 3")
    if nominal_slots < 1 or core_slots < 1:
        raise ValueError("nominal_slots and core_slots must be positive")
    if nominal_slots + core_slots >= schedule_length:
        raise ValueError("the schedule must contain at least one expanded slot")

    env_ids = _resolve_env_ids(env, env_ids)
    num_resets = len(env_ids)
    if num_resets == 0:
        return

    reset_count = _stage2_buffer(
        env,
        _S2_RESET_COUNT_BUFFER,
        shape=(),
        dtype=torch.long,
    )
    local_count = reset_count[env_ids]
    schedule_slot = (local_count + env_ids).remainder(schedule_length)
    is_nominal = schedule_slot < nominal_slots
    is_core = (
        (schedule_slot >= nominal_slots)
        & (schedule_slot < nominal_slots + core_slots)
    )
    is_expanded = ~(is_nominal | is_core)
    spawn_group = torch.full(
        (num_resets,),
        _S2_GROUP_EXPANDED_10MM,
        device=env.device,
        dtype=torch.long,
    )
    spawn_group[is_nominal] = _S2_GROUP_NOMINAL
    spawn_group[is_core] = _S2_GROUP_CORE_5MM

    core_count = _stage2_buffer(
        env,
        _S2_CORE_RESET_COUNT_BUFFER,
        shape=(),
        dtype=torch.long,
    )
    expanded_count = _stage2_buffer(
        env,
        _S2_EXPANDED_RESET_COUNT_BUFFER,
        shape=(),
        dtype=torch.long,
    )
    group_count = torch.where(
        is_core,
        core_count[env_ids],
        expanded_count[env_ids],
    )
    num_strata = strata_per_axis * strata_per_axis
    # Seven is coprime to 25. It spreads parallel environments across all
    # cells; the group-specific counters then cycle every environment through
    # every cell independently for core and expanded samples.
    stratum = (7 * env_ids + group_count).remainder(num_strata)
    stratum_x = stratum.remainder(strata_per_axis)
    stratum_y = torch.div(stratum, strata_per_axis, rounding_mode="floor")
    half_range = torch.where(
        is_core,
        torch.full_like(stratum, core_half_range_m, dtype=torch.float32),
        torch.full_like(stratum, expanded_half_range_m, dtype=torch.float32),
    )
    cell_width = 2.0 * half_range / strata_per_axis
    x_low = -half_range + stratum_x.to(torch.float32) * cell_width
    y_low = -half_range + stratum_y.to(torch.float32) * cell_width

    offsets = torch.zeros((num_resets, 2), device=env.device)
    rejection_count = torch.zeros(num_resets, device=env.device, dtype=torch.long)
    pending = ~is_nominal

    egg = env.scene["egg"]
    default_egg_state = egg.data.default_root_state[env_ids].clone()
    nominal_egg_xy = default_egg_state[:, :2]
    nominal_is_clear = _stage2_spawn_is_clear(
        nominal_egg_xy,
        robot_centerline_x=robot_centerline_x,
        min_robot_centerline_clearance=min_robot_centerline_clearance,
        bucket_xy_local=bucket_xy_local,
        min_bucket_center_clearance=min_bucket_center_clearance,
    )
    if not torch.all(nominal_is_clear):
        raise RuntimeError("The nominal S1 spawn fails the S2-C clearance screen")

    for _ in range(max_resample_attempts):
        if not torch.any(pending):
            break
        sample_count = int(pending.sum().item())
        draw = torch.rand((sample_count, 2), device=env.device)
        candidate_offsets = torch.empty((sample_count, 2), device=env.device)
        candidate_offsets[:, 0] = x_low[pending] + draw[:, 0] * cell_width[pending]
        candidate_offsets[:, 1] = y_low[pending] + draw[:, 1] * cell_width[pending]
        candidate_egg_xy = nominal_egg_xy[pending] + candidate_offsets
        valid = _stage2_spawn_is_clear(
            candidate_egg_xy,
            robot_centerline_x=robot_centerline_x,
            min_robot_centerline_clearance=min_robot_centerline_clearance,
            bucket_xy_local=bucket_xy_local,
            min_bucket_center_clearance=min_bucket_center_clearance,
        )
        pending_indices = pending.nonzero(as_tuple=False).squeeze(-1)
        accepted_indices = pending_indices[valid]
        offsets[accepted_indices] = candidate_offsets[valid]
        rejection_count[pending_indices[~valid]] += 1
        pending[accepted_indices] = False

    if torch.any(pending):
        failed_groups = torch.unique(spawn_group[pending]).tolist()
        failed_strata = torch.unique(stratum[pending]).tolist()
        raise RuntimeError(
            "No collision-clear S2-C spawn found after "
            f"{max_resample_attempts} attempts in groups {failed_groups}, "
            f"strata {failed_strata}. Do not train this distribution."
        )

    egg_state = default_egg_state
    egg_state[:, :3] += env.scene.env_origins[env_ids]
    egg_state[:, :2] += offsets
    egg.write_root_state_to_sim(egg_state, env_ids=env_ids)

    pedestal = env.scene["pedestal"]
    if not pedestal.is_mocap:
        raise RuntimeError("S2-C pedestal must be configured as a mocap body")
    default_pedestal_state = pedestal.data.default_root_state[env_ids]
    pedestal_pose = torch.empty((num_resets, 7), device=env.device)
    pedestal_pose[:, :3] = (
        default_pedestal_state[:, :3] + env.scene.env_origins[env_ids]
    )
    pedestal_pose[:, :2] += offsets
    pedestal_pose[:, 3:7] = default_pedestal_state[:, 3:7]
    pedestal.write_mocap_pose_to_sim(pedestal_pose, env_ids=env_ids)

    _stage2_buffer(
        env,
        _S2_SPAWN_OFFSET_BUFFER,
        shape=(2,),
        dtype=torch.float32,
    )[env_ids] = offsets
    _stage2_buffer(
        env,
        _S2_SPAWN_NOMINAL_BUFFER,
        shape=(),
        dtype=torch.bool,
    )[env_ids] = is_nominal
    _stage2_buffer(
        env,
        _S2_SPAWN_GROUP_BUFFER,
        shape=(),
        dtype=torch.long,
    )[env_ids] = spawn_group
    _stage2_buffer(
        env,
        _S2_SPAWN_STRATUM_BUFFER,
        shape=(),
        dtype=torch.long,
    )[env_ids] = torch.where(is_nominal, -1, stratum)
    _stage2_buffer(
        env,
        _S2_SPAWN_REJECTION_BUFFER,
        shape=(),
        dtype=torch.long,
    )[env_ids] = rejection_count
    core_count[env_ids[is_core]] += 1
    expanded_count[env_ids[is_expanded]] += 1
    reset_count[env_ids] += 1


def reset_stage2_sector_egg_and_pedestal(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    radius_range_m: tuple[float, float],
    angle_range_deg: tuple[float, float],
    radial_strata: int = 5,
    angular_strata: int = 5,
    schedule_length: int = 10,
    nominal_slots: int = 1,
    retention_slots: int = 3,
    max_resample_attempts: int = 128,
    robot_centerline_x: float = 0.0,
    min_robot_centerline_clearance: float = 0.038,
    bucket_xy_local: tuple[float, float] = (-0.05, 0.15),
    min_bucket_center_clearance: float = 0.055,
) -> None:
    """Reset the shared 10/30/60 nominal/retention/sector curriculum.

    Sector samples are uniform in area within balanced radial/angular strata.
    Collision-invalid candidates are rejected and resampled inside the same
    stratum.  Retention samples are uniform over the 38 cells supported by the
    archived S2-C map.  That map is rehearsal material only: the acquisition
    distribution expands through the configured robot-centred polar sector.
    """
    if radial_strata < 1 or angular_strata < 1:
        raise ValueError("radial_strata and angular_strata must be positive")
    if not 0.0 < radius_range_m[0] < radius_range_m[1]:
        raise ValueError("radius_range_m must be positive and ordered")
    if not angle_range_deg[0] < angle_range_deg[1]:
        raise ValueError("angle_range_deg must be ordered")
    if schedule_length < 3:
        raise ValueError("schedule_length must be at least 3")
    if nominal_slots < 1 or retention_slots < 1:
        raise ValueError("nominal_slots and retention_slots must be positive")
    if nominal_slots + retention_slots >= schedule_length:
        raise ValueError("the schedule must contain at least one sector slot")

    env_ids = _resolve_env_ids(env, env_ids)
    num_resets = len(env_ids)
    if num_resets == 0:
        return

    reset_count = _stage2_buffer(
        env,
        _S2_RESET_COUNT_BUFFER,
        shape=(),
        dtype=torch.long,
    )
    local_count = reset_count[env_ids]
    schedule_slot = (local_count + env_ids).remainder(schedule_length)
    is_nominal = schedule_slot < nominal_slots
    is_retention = (
        (schedule_slot >= nominal_slots)
        & (schedule_slot < nominal_slots + retention_slots)
    )
    is_sector = ~(is_nominal | is_retention)
    spawn_group = torch.full(
        (num_resets,),
        _S2_GROUP_SECTOR,
        device=env.device,
        dtype=torch.long,
    )
    spawn_group[is_nominal] = _S2_GROUP_NOMINAL
    spawn_group[is_retention] = _S2_GROUP_RETENTION

    retention_count = _stage2_buffer(
        env,
        _S2_RETENTION_RESET_COUNT_BUFFER,
        shape=(),
        dtype=torch.long,
    )
    sector_count = _stage2_buffer(
        env,
        _S2_SECTOR_RESET_COUNT_BUFFER,
        shape=(),
        dtype=torch.long,
    )

    retention_cells = torch.tensor(
        RETENTION_CELL_LOWER_LEFT_MM,
        device=env.device,
        dtype=torch.float32,
    )
    retention_cell = (env_ids + retention_count[env_ids]).remainder(
        len(RETENTION_CELL_LOWER_LEFT_MM)
    )

    num_sector_strata = radial_strata * angular_strata
    sector_stratum = (env_ids + sector_count[env_ids]).remainder(
        num_sector_strata
    )
    radial_stratum = sector_stratum.remainder(radial_strata)
    angular_stratum = torch.div(
        sector_stratum, radial_strata, rounding_mode="floor"
    )

    radius_min, radius_max = radius_range_m
    radius_width = (radius_max - radius_min) / radial_strata
    radius_low = radius_min + radial_stratum.to(torch.float32) * radius_width
    radius_high = radius_low + radius_width
    angle_min, angle_max = angle_range_deg
    angle_width = (angle_max - angle_min) / angular_strata
    angle_low = angle_min + angular_stratum.to(torch.float32) * angle_width

    offsets = torch.zeros((num_resets, 2), device=env.device)
    rejection_count = torch.zeros(num_resets, device=env.device, dtype=torch.long)
    pending = ~is_nominal

    egg = env.scene["egg"]
    default_egg_state = egg.data.default_root_state[env_ids].clone()
    nominal_egg_xy = default_egg_state[:, :2].clone()
    nominal_is_clear = _stage2_spawn_is_clear(
        nominal_egg_xy,
        robot_centerline_x=robot_centerline_x,
        min_robot_centerline_clearance=min_robot_centerline_clearance,
        bucket_xy_local=bucket_xy_local,
        min_bucket_center_clearance=min_bucket_center_clearance,
    )
    if not torch.all(nominal_is_clear):
        raise RuntimeError("The nominal S1 spawn fails the sector clearance screen")

    for _ in range(max_resample_attempts):
        if not torch.any(pending):
            break
        pending_indices = pending.nonzero(as_tuple=False).squeeze(-1)
        draw = torch.rand((len(pending_indices), 2), device=env.device)
        candidate_xy = torch.empty((len(pending_indices), 2), device=env.device)

        pending_retention = is_retention[pending_indices]
        if torch.any(pending_retention):
            indices = pending_indices[pending_retention]
            cell_low_mm = retention_cells[retention_cell[indices]]
            within_cell = draw[pending_retention] * RETENTION_CELL_SIZE_MM
            candidate_offset_m = (cell_low_mm + within_cell) / 1000.0
            candidate_xy[pending_retention] = nominal_egg_xy[indices] + (
                candidate_offset_m
            )

        pending_sector = is_sector[pending_indices]
        if torch.any(pending_sector):
            indices = pending_indices[pending_sector]
            sector_draw = draw[pending_sector]
            r_low = radius_low[indices]
            r_high = radius_high[indices]
            radius = torch.sqrt(
                r_low.square()
                + sector_draw[:, 0] * (r_high.square() - r_low.square())
            )
            angle_deg = angle_low[indices] + sector_draw[:, 1] * angle_width
            angle_rad = torch.deg2rad(angle_deg)
            candidate_xy[pending_sector, 0] = radius * torch.cos(angle_rad)
            candidate_xy[pending_sector, 1] = radius * torch.sin(angle_rad)

        valid = _stage2_spawn_is_clear(
            candidate_xy,
            robot_centerline_x=robot_centerline_x,
            min_robot_centerline_clearance=min_robot_centerline_clearance,
            bucket_xy_local=bucket_xy_local,
            min_bucket_center_clearance=min_bucket_center_clearance,
        )
        accepted_indices = pending_indices[valid]
        offsets[accepted_indices] = (
            candidate_xy[valid] - nominal_egg_xy[accepted_indices]
        )
        rejection_count[pending_indices[~valid]] += 1
        pending[accepted_indices] = False

    if torch.any(pending):
        failed_groups = torch.unique(spawn_group[pending]).tolist()
        failed_strata = torch.unique(sector_stratum[pending & is_sector]).tolist()
        raise RuntimeError(
            "No collision-clear sector spawn found after "
            f"{max_resample_attempts} attempts in groups {failed_groups}, "
            f"sector strata {failed_strata}."
        )

    egg_state = default_egg_state
    egg_state[:, :3] += env.scene.env_origins[env_ids]
    egg_state[:, :2] += offsets
    egg.write_root_state_to_sim(egg_state, env_ids=env_ids)

    pedestal = env.scene["pedestal"]
    if not pedestal.is_mocap:
        raise RuntimeError("Sector pedestal must be configured as a mocap body")
    default_pedestal_state = pedestal.data.default_root_state[env_ids]
    pedestal_pose = torch.empty((num_resets, 7), device=env.device)
    pedestal_pose[:, :3] = (
        default_pedestal_state[:, :3] + env.scene.env_origins[env_ids]
    )
    pedestal_pose[:, :2] += offsets
    pedestal_pose[:, 3:7] = default_pedestal_state[:, 3:7]
    pedestal.write_mocap_pose_to_sim(pedestal_pose, env_ids=env_ids)

    spawn_xy = nominal_egg_xy + offsets
    spawn_radius = torch.linalg.norm(spawn_xy, dim=-1)
    spawn_angle_deg = torch.rad2deg(torch.atan2(spawn_xy[:, 1], spawn_xy[:, 0]))
    reported_radial_stratum = torch.where(
        is_sector, radial_stratum, torch.full_like(radial_stratum, -1)
    )
    reported_angular_stratum = torch.where(
        is_sector, angular_stratum, torch.full_like(angular_stratum, -1)
    )
    reported_stratum = torch.where(
        is_sector,
        sector_stratum,
        torch.where(is_retention, retention_cell, -1),
    )

    _stage2_buffer(
        env, _S2_SPAWN_OFFSET_BUFFER, shape=(2,), dtype=torch.float32
    )[env_ids] = offsets
    _stage2_buffer(
        env, _S2_SPAWN_NOMINAL_BUFFER, shape=(), dtype=torch.bool
    )[env_ids] = is_nominal
    _stage2_buffer(
        env, _S2_SPAWN_GROUP_BUFFER, shape=(), dtype=torch.long
    )[env_ids] = spawn_group
    _stage2_buffer(
        env, _S2_SPAWN_STRATUM_BUFFER, shape=(), dtype=torch.long
    )[env_ids] = reported_stratum
    _stage2_buffer(
        env, _S2_SPAWN_REJECTION_BUFFER, shape=(), dtype=torch.long
    )[env_ids] = rejection_count
    _stage2_buffer(
        env, _S2_SPAWN_RADIUS_BUFFER, shape=(), dtype=torch.float32
    )[env_ids] = spawn_radius
    _stage2_buffer(
        env, _S2_SPAWN_ANGLE_BUFFER, shape=(), dtype=torch.float32
    )[env_ids] = spawn_angle_deg
    _stage2_buffer(
        env, _S2_SPAWN_RADIAL_STRATUM_BUFFER, shape=(), dtype=torch.long
    )[env_ids] = reported_radial_stratum
    _stage2_buffer(
        env, _S2_SPAWN_ANGULAR_STRATUM_BUFFER, shape=(), dtype=torch.long
    )[env_ids] = reported_angular_stratum

    retention_count[env_ids[is_retention]] += 1
    sector_count[env_ids[is_sector]] += 1
    reset_count[env_ids] += 1


def stage2_spawn_offset_x_mm(env: "ManagerBasedRlEnv") -> torch.Tensor:
    offsets = _stage2_buffer(
        env,
        _S2_SPAWN_OFFSET_BUFFER,
        shape=(2,),
        dtype=torch.float32,
    )
    return offsets[:, 0] * 1000.0


def stage2_spawn_offset_y_mm(env: "ManagerBasedRlEnv") -> torch.Tensor:
    offsets = _stage2_buffer(
        env,
        _S2_SPAWN_OFFSET_BUFFER,
        shape=(2,),
        dtype=torch.float32,
    )
    return offsets[:, 1] * 1000.0


def stage2_spawn_abs_offset_x_mm(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return torch.abs(stage2_spawn_offset_x_mm(env))


def stage2_spawn_abs_offset_y_mm(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return torch.abs(stage2_spawn_offset_y_mm(env))


def stage2_spawn_is_nominal(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _stage2_buffer(
        env,
        _S2_SPAWN_NOMINAL_BUFFER,
        shape=(),
        dtype=torch.bool,
    ).to(torch.float32)


def stage2_spawn_stratum(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _stage2_buffer(
        env,
        _S2_SPAWN_STRATUM_BUFFER,
        shape=(),
        dtype=torch.long,
    ).to(torch.float32)


def stage2_spawn_is_core_5mm(env: "ManagerBasedRlEnv") -> torch.Tensor:
    groups = _stage2_buffer(
        env,
        _S2_SPAWN_GROUP_BUFFER,
        shape=(),
        dtype=torch.long,
    )
    return (groups == _S2_GROUP_CORE_5MM).to(torch.float32)


def stage2_spawn_is_expanded_10mm(env: "ManagerBasedRlEnv") -> torch.Tensor:
    groups = _stage2_buffer(
        env,
        _S2_SPAWN_GROUP_BUFFER,
        shape=(),
        dtype=torch.long,
    )
    return (groups == _S2_GROUP_EXPANDED_10MM).to(torch.float32)


def stage2_spawn_is_retention(env: "ManagerBasedRlEnv") -> torch.Tensor:
    groups = _stage2_buffer(
        env, _S2_SPAWN_GROUP_BUFFER, shape=(), dtype=torch.long
    )
    return (groups == _S2_GROUP_RETENTION).to(torch.float32)


def stage2_spawn_is_sector(env: "ManagerBasedRlEnv") -> torch.Tensor:
    groups = _stage2_buffer(
        env, _S2_SPAWN_GROUP_BUFFER, shape=(), dtype=torch.long
    )
    return (groups == _S2_GROUP_SECTOR).to(torch.float32)


def stage2_spawn_radius_mm(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _stage2_buffer(
        env, _S2_SPAWN_RADIUS_BUFFER, shape=(), dtype=torch.float32
    ) * 1000.0


def stage2_spawn_angle_deg(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _stage2_buffer(
        env, _S2_SPAWN_ANGLE_BUFFER, shape=(), dtype=torch.float32
    )


def stage2_spawn_radial_stratum(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _stage2_buffer(
        env, _S2_SPAWN_RADIAL_STRATUM_BUFFER, shape=(), dtype=torch.long
    ).to(torch.float32)


def stage2_spawn_angular_stratum(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _stage2_buffer(
        env, _S2_SPAWN_ANGULAR_STRATUM_BUFFER, shape=(), dtype=torch.long
    ).to(torch.float32)


def stage2_spawn_rejection_count(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _stage2_buffer(
        env,
        _S2_SPAWN_REJECTION_BUFFER,
        shape=(),
        dtype=torch.long,
    ).to(torch.float32)


def tendon_length(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = ROBOT_CFG) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    return robot.data.tendon_len[:, asset_cfg.tendon_ids]


def tendon_velocity(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = ROBOT_CFG) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    return robot.data.tendon_vel[:, asset_cfg.tendon_ids]


def egg_position(env: "ManagerBasedRlEnv", object_name: str = "egg") -> torch.Tensor:
    obj: Entity = env.scene[object_name]
    return obj.data.root_link_pos_w


def bucket_position(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = BUCKET_CFG) -> torch.Tensor:
    bucket: Entity = env.scene[asset_cfg.name]
    return bucket.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)


def egg_to_bucket(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = BUCKET_CFG) -> torch.Tensor:
    return bucket_position(env, asset_cfg) - egg_position(env)


def touch_values(env: "ManagerBasedRlEnv", n_sensors: int = 42, threshold: float = 1.0e-4) -> torch.Tensor:
    """Binary robot touch sensors from MuJoCo sensordata."""
    raw = env.sim.data.sensordata[:, :n_sensors]
    return (raw > threshold).float()


def last_action(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return env.action_manager.action

def egg_inside_bucket(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = BUCKET_CFG,
    com_distance_threshold: float = 0.015,
    min_z_offset: float = -0.010,
    max_z_offset: float = 0.020,
) -> torch.Tensor:
    """Strict bucket success detector.

    Success requires the egg root/COM to be close to bucket_site and vertically
    inside the bucket interior, not merely near the rim.
    """
    egg = egg_position(env)
    bucket = bucket_position(env, asset_cfg)
    delta = egg - bucket
    com_close = torch.linalg.norm(delta, dim=-1) < com_distance_threshold
    z_inside = (delta[:, 2] > min_z_offset) & (delta[:, 2] < max_z_offset)
    return com_close & z_inside

def egg_inside_bucket_terminal_indicator(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = BUCKET_CFG,
    com_distance_threshold: float = 0.015,
    max_z_offset: float = 0.02,
) -> torch.Tensor:
    """Return a one-shot successful-placement indicator."""
    success = egg_inside_bucket(
        env,
        asset_cfg=asset_cfg,
        com_distance_threshold=com_distance_threshold,
        max_z_offset=max_z_offset,
    )
    return success.to(dtype=torch.float32) / env.step_dt

def egg_fell(env: "ManagerBasedRlEnv", min_z: float = 0.03) -> torch.Tensor:
    """Safety termination if the egg falls below the useful workspace."""
    return egg_position(env)[:, 2] < min_z

def egg_fell_terminal_indicator(env: "ManagerBasedRlEnv",) -> torch.Tensor:
    """Return a one-shot egg-fall indicator.

    Reward terms are integrated over the control timestep. Dividing by
    step_dt makes the configured weight behave as an actual terminal amount.
    """
    return egg_fell(env).to(dtype=torch.float32) / env.step_dt

def _world_xy_to_env_local(
    env: "ManagerBasedRlEnv",
    position_w: torch.Tensor,
) -> torch.Tensor:
    """Convert batched world-frame XY positions to environment-local XY."""
    return position_w[:, :2] - env.scene.env_origins[:, :2]

def egg_out_of_bounds(env: "ManagerBasedRlEnv", xy_limit: float = 0.40) -> torch.Tensor:
    """Terminate if the egg escapes the small manipulation workspace."""
    egg_xy_local = _world_xy_to_env_local(env, egg_position(env),)
    return torch.any(torch.abs(egg_xy_local) > xy_limit, dim=-1,)


def nan_state(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return ~torch.isfinite(env.sim.data.qacc).all(dim=-1)


def egg_to_bucket_delta_progress(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = BUCKET_CFG,
    progress_scale: float = 0.005,
) -> torch.Tensor:
    """Reward the signed step-to-step reduction in egg-to-bucket distance.

    A net 5 mm movement toward the bucket gives approximately +1 total
    reward across the corresponding control steps. Moving away produces
    a negative reward. Holding the egg stationary produces zero.
    """

    egg_xy = (
        egg_position(env)[:, :2]
        - env.scene.env_origins[:, :2]
    )

    bucket_xy = (
        bucket_position(env, asset_cfg)[:, :2]
        - env.scene.env_origins[:, :2]
    )

    current_distance = torch.linalg.norm(
        egg_xy - bucket_xy,
        dim=-1,
    )

    buffer_name = "_stage1_previous_egg_bucket_distance"

    previous_distance = getattr(env, buffer_name, None)

    if (
        previous_distance is None
        or previous_distance.shape != current_distance.shape
    ):
        setattr(
            env,
            buffer_name,
            current_distance.detach().clone(),
        )
        return torch.zeros_like(current_distance)

    progress = previous_distance - current_distance

    # Do not interpret a newly reset environment as object movement.
    new_episode = env.episode_length_buf <= 1
    progress = torch.where(
        new_episode,
        torch.zeros_like(progress),
        progress,
    )

    setattr(
        env,
        buffer_name,
        current_distance.detach().clone(),
    )

    # RewardManager multiplies terms by step_dt. Dividing here makes the
    # final contribution equal to the intended normalized progress.
    normalized_progress = progress / progress_scale

    return torch.clamp(
        normalized_progress,
        min=-1.0,
        max=1.0,
    ) / env.step_dt

def robot_keypoints_xy(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """XY positions of SpiRob link/joint keypoints in the local environment frame.

    This represents the visible shape of the robot and can later be
    estimated from a top-view camera.
    """

    robot: Entity = env.scene[asset_cfg.name]

    # [num_envs, num_links, 2]
    keypoints_w = robot.data.body_link_pos_w[
        :,
        asset_cfg.body_ids,
        :2,
    ]

    # Convert world coordinates to each parallel environment's local frame.
    keypoints_local = (
        keypoints_w
        - env.scene.env_origins[:, None, :2]
    )

    # [num_envs, num_links * 2]
    return keypoints_local.flatten(start_dim=1)
