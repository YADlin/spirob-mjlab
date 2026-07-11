"""Minimal SpiRob MDP terms for mjlab.

The terms are intentionally simple and vectorized over all mjlab environments.
They are the mjlab equivalent of the user's first Gym proof-of-concept:
  tendon state + touch sensors + tip-to-egg vector -> reach/contact reward.
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


def touch_values(env: "ManagerBasedRlEnv", n_sensors: int = 20, threshold: float = 1.0e-4) -> torch.Tensor:
    """Binary robot touch sensors from MuJoCo sensordata.

    The robot XML contributes 20 <touch> sensors. This returns [num_envs, 20].
    """
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
    n_sensors: int = 20,
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
    n_sensors: int = 20,
    touch_threshold: float = 1.0e-4,
) -> torch.Tensor:
    """Terminate as success when close enough or when any touch sensor fires."""
    d = torch.linalg.norm(tip_to_egg(env, asset_cfg), dim=-1)
    touched = touch_values(env, n_sensors=n_sensors, threshold=touch_threshold).sum(dim=-1) > 0
    return (d < distance_threshold) | touched


def egg_fell(env: "ManagerBasedRlEnv", min_z: float = 0.03) -> torch.Tensor:
    """Safety termination for early debugging."""
    return egg_position(env)[:, 2] < min_z


def nan_state(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return ~torch.isfinite(env.sim.data.qacc).all(dim=-1)

def egg_to_bucket_distance(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = BUCKET_CFG,
) -> torch.Tensor:
    """3-D distance from egg root to bucket target site."""
    d = egg_to_bucket(env, asset_cfg)
    return torch.linalg.norm(d, dim=-1)


def egg_xy_distance_to_bucket(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = BUCKET_CFG,
) -> torch.Tensor:
    """XY distance from egg root to bucket target site."""
    d = egg_to_bucket(env, asset_cfg)
    return torch.linalg.norm(d[:, :2], dim=-1)


def egg_to_bucket_reward(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = BUCKET_CFG,
    std: float = 0.05,
) -> torch.Tensor:
    """Smooth shaping reward for bucket-drop: 1 at bucket_site, ~0 far away."""
    d = egg_to_bucket(env, asset_cfg)
    return torch.exp(-torch.sum(d * d, dim=-1) / (std * std))


def egg_inside_bucket(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = BUCKET_CFG,
    xy_threshold: float = 0.026,
    min_z_offset: float = -0.010,
    max_z_offset: float = 0.055,
) -> torch.Tensor:
    """Success detector for the simple box-bucket.

    The bucket target site is at the approximate cup center. We declare success
    when the egg root is horizontally inside the bucket and within a vertical
    band around the bucket interior. This is intentionally geometry-based rather
    than contact-pair-based so it remains cheap and vectorized over all envs.
    """
    egg = egg_position(env)
    bucket = bucket_position(env, asset_cfg)
    delta = egg - bucket
    xy_ok = torch.linalg.norm(delta[:, :2], dim=-1) < xy_threshold
    z_ok = (delta[:, 2] > min_z_offset) & (delta[:, 2] < max_z_offset)
    return xy_ok & z_ok


def egg_inside_bucket_reward(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = BUCKET_CFG,
) -> torch.Tensor:
    """Float version of egg_inside_bucket for RewardTermCfg."""
    return egg_inside_bucket(env, asset_cfg).float()


def egg_missed_bucket(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = BUCKET_CFG,
    miss_z_offset: float = -0.005,
    xy_fail_threshold: float = 0.045,
) -> torch.Tensor:
    """Terminate failed drops once the egg has fallen below the bucket center.

    If the egg is already below the target site but outside a generous XY radius,
    it has missed the bucket. This prevents wasting the full episode on obvious
    failed drops while avoiding false failure during the initial fall.
    """
    egg = egg_position(env)
    bucket = bucket_position(env, asset_cfg)
    delta = egg - bucket
    below_cup_center = delta[:, 2] < miss_z_offset
    outside_bucket = torch.linalg.norm(delta[:, :2], dim=-1) > xy_fail_threshold
    return below_cup_center & outside_bucket

