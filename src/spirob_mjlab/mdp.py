"""SpiRob MDP terms for mjlab.

The functions are vectorized over all mjlab environments. Stage 1 manipulation
uses the smallest meaningful reward: reduce egg-to-bucket distance, with a
sparse terminal bonus when the egg is actually inside the bucket.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

ROBOT_CFG = SceneEntityCfg(
    "robot",
    tendon_names=("cable_0", "cable_1"),
    site_names=("tip_site",),
)
BUCKET_CFG = SceneEntityCfg("bucket", site_names=("bucket_site",))


def tendon_length(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = ROBOT_CFG) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    return robot.data.tendon_len[:, asset_cfg.tendon_ids]


def tendon_velocity(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = ROBOT_CFG) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    return robot.data.tendon_vel[:, asset_cfg.tendon_ids]


def tip_position(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = ROBOT_CFG) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    return robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)


def egg_position(env: "ManagerBasedRlEnv", object_name: str = "egg") -> torch.Tensor:
    obj: Entity = env.scene[object_name]
    return obj.data.root_link_pos_w


def bucket_position(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = BUCKET_CFG) -> torch.Tensor:
    bucket: Entity = env.scene[asset_cfg.name]
    return bucket.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)


def tip_to_egg(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = ROBOT_CFG) -> torch.Tensor:
    return egg_position(env) - tip_position(env, asset_cfg)


def egg_to_bucket(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = BUCKET_CFG) -> torch.Tensor:
    return bucket_position(env, asset_cfg) - egg_position(env)


def touch_values(env: "ManagerBasedRlEnv", n_sensors: int = 42, threshold: float = 1.0e-4) -> torch.Tensor:
    """Binary robot touch sensors from MuJoCo sensordata."""
    raw = env.sim.data.sensordata[:, :n_sensors]
    return (raw > threshold).float()


def last_action(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return env.action_manager.action


def reach_reward(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = ROBOT_CFG,
    std: float = 0.06,
) -> torch.Tensor:
    """Smooth reach reward, 1 near the egg and ~0 far away."""
    d = tip_to_egg(env, asset_cfg)
    return torch.exp(-torch.sum(d * d, dim=-1) / (std * std))


def contact_reward(
    env: "ManagerBasedRlEnv",
    n_sensors: int = 42,
    threshold: float = 1.0e-4,
) -> torch.Tensor:
    """Small reward proportional to the number of active touch sensors."""
    return touch_values(env, n_sensors=n_sensors, threshold=threshold).mean(dim=-1)


def action_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Positive L2 action cost; give it a negative RewardTermCfg weight."""
    a = env.action_manager.action
    return torch.sum(a * a, dim=-1)


def reached_egg(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = ROBOT_CFG,
    distance_threshold: float = 0.012,
    n_sensors: int = 42,
    touch_threshold: float = 1.0e-4,
) -> torch.Tensor:
    """Terminate as success when close enough or when any touch sensor fires."""
    d = torch.linalg.norm(tip_to_egg(env, asset_cfg), dim=-1)
    touched = touch_values(env, n_sensors=n_sensors, threshold=touch_threshold).sum(dim=-1) > 0
    return (d < distance_threshold) | touched


def egg_to_bucket_distance(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = BUCKET_CFG,
) -> torch.Tensor:
    """3D distance from egg root/COM to bucket target site."""
    return torch.linalg.norm(egg_to_bucket(env, asset_cfg), dim=-1)


def egg_to_bucket_xy_distance(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = BUCKET_CFG,
) -> torch.Tensor:
    """XY distance from egg root/COM to bucket target site."""
    return torch.linalg.norm(egg_to_bucket(env, asset_cfg)[:, :2], dim=-1)


def egg_to_bucket_distance_reward(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = BUCKET_CFG,
    distance_scale: float = 0.10,
) -> torch.Tensor:
    """Minimal dense manipulation reward.

    Returns a larger value as the egg approaches the bucket. This deliberately
    ignores whether the robot touched the egg. At this stage, we want to observe
    whether random tendon exploration can ever move the object in the right
    direction. If it cannot, the next reward term to add is reach/contact.
    """
    d = egg_to_bucket_distance(env, asset_cfg)
    return -d / distance_scale


def egg_to_bucket_smooth_reward(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = BUCKET_CFG,
    std: float = 0.08,
) -> torch.Tensor:
    """Optional bounded version of the distance reward, useful for diagnostics."""
    d = egg_to_bucket(env, asset_cfg)
    return torch.exp(-torch.sum(d * d, dim=-1) / (std * std))


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


def egg_inside_bucket_reward(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = BUCKET_CFG,
    com_distance_threshold: float = 0.015,
    min_z_offset: float = -0.010,
    max_z_offset: float = 0.020,
) -> torch.Tensor:
    """Float version of egg_inside_bucket for RewardTermCfg."""
    return egg_inside_bucket(
        env,
        asset_cfg=asset_cfg,
        com_distance_threshold=com_distance_threshold,
        min_z_offset=min_z_offset,
        max_z_offset=max_z_offset,
    ).float()


def egg_fell(env: "ManagerBasedRlEnv", min_z: float = 0.03) -> torch.Tensor:
    """Safety termination if the egg falls below the useful workspace."""
    return egg_position(env)[:, 2] < min_z

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

def egg_directed_progress_from_spawn(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = BUCKET_CFG,
    egg_spawn_xy: tuple[float, float] = (0.05, 0.15),
    progress_scale: float = 0.005,
    max_progress: float = 0.10,
) -> torch.Tensor:
    """Reward egg XY movement specifically toward the bucket.

    This is deterministic Stage-1 shaping:
    - It does not reward touching.
    - It does not reward random displacement.
    - It gives a strong reward for the first few millimetres of egg motion
      in the bucket direction.
    """
    egg_xy_local = (egg_position(env)[:, :2] - env.scene.env_origins[:, :2])
    bucket_xy_local = (bucket_position(env, asset_cfg)[:, :2]- env.scene.env_origins[:, :2])

    spawn_xy = torch.tensor(
        egg_spawn_xy,
        device=egg_xy_local.device,
        dtype=egg_xy_local.dtype,
    ).unsqueeze(0)

    direction = bucket_xy_local - spawn_xy
    direction = direction / torch.clamp(
        torch.linalg.norm(direction, dim=-1, keepdim=True),
        min=1.0e-6,
    )

    displacement = egg_xy_local - spawn_xy
    progress = torch.sum(displacement * direction, dim=-1)

    progress = torch.clamp(progress, min=0.0, max=max_progress)
    return progress / progress_scale

def egg_first_push_bonus(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = BUCKET_CFG,
    egg_spawn_xy: tuple[float, float] = (0.05, 0.15),
    push_scale: float = 0.002,
) -> torch.Tensor:
    """Saturating bonus for the first useful egg displacement.

    A 1-2 mm push toward the bucket becomes noticeable.
    The exponential saturates, so it does not dominate the whole task forever.
    """
    egg_xy_local = (egg_position(env)[:, :2] - env.scene.env_origins[:, :2])
    bucket_xy_local = (bucket_position(env, asset_cfg)[:, :2]- env.scene.env_origins[:, :2])

    spawn_xy = torch.tensor(
        egg_spawn_xy,
        device=egg_xy_local.device,
        dtype=egg_xy_local.dtype,
    ).unsqueeze(0)

    direction = bucket_xy_local - spawn_xy
    direction = direction / torch.clamp(
        torch.linalg.norm(direction, dim=-1, keepdim=True),
        min=1.0e-6,
    )

    displacement = egg_xy_local - spawn_xy
    progress = torch.sum(displacement * direction, dim=-1)
    progress = torch.clamp(progress, min=0.0)

    return 1.0 - torch.exp(-progress / push_scale)