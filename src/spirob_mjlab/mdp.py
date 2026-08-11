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


def action_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Positive L2 action cost; give it a negative RewardTermCfg weight."""
    a = env.action_manager.action
    return torch.sum(a * a, dim=-1)


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

