"""MDP terms for the SpiRob egg-knockdown task.

Actor information is limited to the camera/encoder contract (``contract.py``).
Everything else here (true egg state, robot keypoints, contact) is privileged
and used only for rewards, terminations and the separate critic.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
# math is used for the default yaw range below.

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from spirob_mjlab.knockdown import contract as C

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

ACTION_TERM = "cable_len"
CONTACT_SENSOR = "egg_robot_contact"
EGG_SETTLED_Z = 0.0950          # egg root z resting on the pedestal (measured, CPU MuJoCo 3.8.1)
PEDESTAL_ROOT_Z = 0.002


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _action(env: "ManagerBasedRlEnv"):
    return env.action_manager.get_term(ACTION_TERM)


def _c_origin_w(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Frame-C origin in world XY for every env (robot entity is at env origin)."""
    o = torch.tensor(C.C_ORIGIN_MODEL_XY, device=env.device)
    return env.scene.env_origins[:, :2] + o


def world_xy_to_c(env: "ManagerBasedRlEnv", xy_w: torch.Tensor) -> torch.Tensor:
    """(..., 2) world XY -> frame C (u, v); leading dim is num_envs."""
    origin = _c_origin_w(env)
    shape = (-1,) + (1,) * (xy_w.dim() - 2) + (2,)
    rel = xy_w - origin.view(shape)
    return torch.stack((rel[..., 0], -rel[..., 1]), dim=-1)


def c_to_world_xy(env: "ManagerBasedRlEnv", uv: torch.Tensor) -> torch.Tensor:
    origin = _c_origin_w(env)
    return torch.stack((origin[:, 0] + uv[:, 0], origin[:, 1] - uv[:, 1]), dim=-1)


def egg_pos_w(env: "ManagerBasedRlEnv") -> torch.Tensor:
    egg: Entity = env.scene["egg"]
    return egg.data.root_link_pos_w


def pedestal_pos_w(env: "ManagerBasedRlEnv") -> torch.Tensor:
    ped: Entity = env.scene["pedestal"]
    return ped.data.root_link_pos_w


class KnockdownState:
    """Per-env episode buffers shared by observation/reward/termination terms."""

    def __init__(self, env: "ManagerBasedRlEnv"):
        n, dev = env.num_envs, env.device
        self.cam_bias = torch.zeros(n, 2, device=dev)
        self.egg_uv_start_obs = torch.zeros(n, 2, device=dev)
        self.ped_uv = torch.zeros(n, 2, device=dev)


def state(env: "ManagerBasedRlEnv") -> KnockdownState:
    s = getattr(env, "_knockdown_state", None)
    if s is None:
        s = KnockdownState(env)
        env._knockdown_state = s
    return s


# ---------------------------------------------------------------------------
# Camera model (sim stand-in for lab/rl/egg_vision.py)
# ---------------------------------------------------------------------------
def camera_egg_uv(env, true_uv: torch.Tensor, noise_std_m: float) -> torch.Tensor:
    s = state(env)
    return true_uv + s.cam_bias + noise_std_m * torch.randn_like(true_uv)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
def sample_region_uv(n: int, device, abs_u_range, v_range, side: str = "both") -> torch.Tensor:
    """Uniform samples in the placement rectangle(s) of frame C.

    |u| : lateral distance from the rest backbone line [m]
    v   : distance along the rest backbone from joint link_002 [m]
    side: 'both' (mirror rectangles), 'positive' (+u) or 'negative' (-u).
    """
    a0, a1 = abs_u_range
    v0, v1 = v_range
    au = a0 + (a1 - a0) * torch.rand(n, device=device)
    v = v0 + (v1 - v0) * torch.rand(n, device=device)
    if side == "both":
        sign = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    else:
        sign = torch.full((n,), 1.0 if side == "positive" else -1.0, device=device)
    return torch.stack((sign * au, v), dim=-1)


def reset_egg_and_pedestal(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    abs_u_range: tuple[float, float],
    v_range: tuple[float, float],
    side: str = "both",
    fixed_uv: tuple[float, float] | None = None,
    cam_bias_m: float = 0.003,
    cam_noise_m: float = 0.001,
    yaw_range_rad: float = math.pi,
) -> None:
    """Place pedestal and egg together (same XY) relative to the robot base."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    if n == 0:
        return
    if fixed_uv is not None:
        uv = torch.tensor(fixed_uv, device=env.device).repeat(n, 1)
    else:
        uv = sample_region_uv(n, env.device, abs_u_range, v_range, side)
    # Evaluation hook: env._kd_uv_override is an (num_envs, 2) tensor; rows that
    # are finite replace the sampled placement for those environments.
    override = getattr(env, "_kd_uv_override", None)
    if override is not None:
        ov = override[env_ids]
        use = torch.isfinite(ov).all(dim=-1)
        uv = torch.where(use[:, None], ov, uv)
    origin = _c_origin_w(env)[env_ids]
    xy = torch.stack((origin[:, 0] + uv[:, 0], origin[:, 1] - uv[:, 1]), dim=-1)

    ped: Entity = env.scene["pedestal"]
    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0:2] = xy
    pose[:, 2] = env.scene.env_origins[env_ids, 2] + PEDESTAL_ROOT_Z
    pose[:, 3] = 1.0
    ped.write_mocap_pose_to_sim(pose, env_ids=env_ids)

    egg: Entity = env.scene["egg"]
    yaw = yaw_range_rad * (2 * torch.rand(n, device=env.device) - 1)
    root = torch.zeros(n, 13, device=env.device)
    root[:, 0:2] = xy
    root[:, 2] = env.scene.env_origins[env_ids, 2] + EGG_SETTLED_Z
    root[:, 3] = torch.cos(yaw / 2)
    root[:, 6] = torch.sin(yaw / 2)
    egg.write_root_state_to_sim(root, env_ids=env_ids)

    s = state(env)
    s.ped_uv[env_ids] = uv
    s.cam_bias[env_ids] = cam_bias_m * (2 * torch.rand(n, 2, device=env.device) - 1)
    s.egg_uv_start_obs[env_ids] = uv + s.cam_bias[env_ids] + cam_noise_m * torch.randn(n, 2, device=env.device)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
def egg_uv_true(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return world_xy_to_c(env, egg_pos_w(env)[:, :2])


def actor_contract_obs(env: "ManagerBasedRlEnv", cam_noise_m: float = 0.001) -> torch.Tensor:
    """The full actor vector, assembled by the shared contract function."""
    act = _action(env)
    s = state(env)
    egg_uv = camera_egg_uv(env, egg_uv_true(env), cam_noise_m)
    return C.build_actor_obs(egg_uv, s.egg_uv_start_obs, act.payout, act.prev_action)


def critic_privileged(env: "ManagerBasedRlEnv", keypoint_cfg: SceneEntityCfg) -> torch.Tensor:
    """Privileged state for the critic only."""
    act = _action(env)
    s = state(env)
    robot: Entity = env.scene["robot"]
    egg_w = egg_pos_w(env)
    egg_uv = world_xy_to_c(env, egg_w[:, :2])
    egg_rel = torch.cat((egg_uv - s.ped_uv, (egg_w[:, 2:3] - env.scene.env_origins[:, 2:3] - EGG_SETTLED_Z)), dim=-1)
    egg_vel = env.scene["egg"].data.root_link_lin_vel_w
    kp = world_xy_to_c(env, robot.data.body_link_pos_w[:, keypoint_cfg.body_ids, :2])
    ten = robot.data.tendon_len[:, :2]
    touched = egg_touched(env).float().unsqueeze(-1)
    clean = C.build_actor_obs(egg_uv, s.ped_uv, act.payout, act.prev_action)
    frac = (env.episode_length_buf.float() / env.max_episode_length).unsqueeze(-1)
    return torch.cat((clean, egg_rel / 0.05, egg_vel.clamp(-2, 2), kp.flatten(1) / 0.1,
                      (ten - torch.tensor(C.NEUTRAL_PAYOUT_M, device=env.device)) / 0.06,
                      act.reference_offset / 0.002, touched, frac), dim=-1)


# ---------------------------------------------------------------------------
# Contact / classification helpers (privileged)
# ---------------------------------------------------------------------------
def egg_touched(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """True if robot and egg have been in contact at any physics step this episode."""
    sensor = env.scene[CONTACT_SENSOR]
    d = sensor.data
    hit = (d.current_contact_time > 0) | (d.last_contact_time > 0)
    return hit.any(dim=-1)


def egg_below_support(env: "ManagerBasedRlEnv", spec: C.SuccessSpec = C.DEFAULT.success) -> torch.Tensor:
    z = egg_pos_w(env)[:, 2] - env.scene.env_origins[:, 2]
    return z < (spec.sim_support_top_z_m - spec.sim_fall_margin_m)


def numerical_fault(env: "ManagerBasedRlEnv") -> torch.Tensor:
    bad_q = ~torch.isfinite(env.sim.data.qacc).all(dim=-1)
    bad_egg = ~torch.isfinite(egg_pos_w(env)).all(dim=-1)
    return bad_q | bad_egg


# Terminations: mutually exclusive by construction, precedence
#   numerical_fault > knocked_down > spontaneous_fall > egg_out_of_bounds > time_out
def term_numerical_fault(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return numerical_fault(env)


def term_knocked_down(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return ~numerical_fault(env) & egg_below_support(env) & egg_touched(env)


def term_spontaneous_fall(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return ~numerical_fault(env) & egg_below_support(env) & ~egg_touched(env)


def term_egg_out_of_bounds(env: "ManagerBasedRlEnv", xy_limit: float = 0.40) -> torch.Tensor:
    rel = egg_pos_w(env)[:, :2] - env.scene.env_origins[:, :2]
    oob = (rel.abs() > xy_limit).any(dim=-1)
    return ~numerical_fault(env) & ~egg_below_support(env) & oob


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
def knockdown_bonus(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """One-time: the episode terminates on the same condition."""
    return term_knocked_down(env).float() / env.step_dt


class _PotentialDelta:
    """Reward = Phi(s_t) - Phi(s_{t-1}); telescopes, so it cannot be farmed."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        self.prev = torch.zeros(env.num_envs, device=env.device)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def potential(self, env, **kwargs) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, env, **kwargs) -> torch.Tensor:
        phi = self.potential(env, **kwargs)
        out = torch.where(self.fresh, torch.zeros_like(phi), phi - self.prev)
        self.prev = phi.detach().clone()
        self.fresh[:] = False
        return out / env.step_dt

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.fresh[env_ids] = True


class approach_progress(_PotentialDelta):
    """Phi = -clip(min robot-keypoint XY distance to the egg, 0, d_max) / d_max."""

    def potential(self, env, keypoint_cfg: SceneEntityCfg, d_max: float = 0.10) -> torch.Tensor:
        robot: Entity = env.scene["robot"]
        kp = robot.data.body_link_pos_w[:, keypoint_cfg.body_ids, :2]
        egg = egg_pos_w(env)[:, None, :2]
        d = torch.linalg.norm(kp - egg, dim=-1).min(dim=-1).values
        return -d.clamp(0.0, d_max) / d_max


class egg_displacement_progress(_PotentialDelta):
    """Phi = clip(egg XY offset from its pedestal, 0, d_max) / d_max."""

    def potential(self, env, d_max: float = 0.03) -> torch.Tensor:
        s = state(env)
        d = torch.linalg.norm(egg_uv_true(env) - s.ped_uv, dim=-1)
        return d.clamp(0.0, d_max) / d_max


class command_motion_cost:
    """Mean |limited target change| / max_step: 1.0 for a full-size move on both cables."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        self.prev = torch.tensor(C.NEUTRAL_PAYOUT_M, device=env.device).repeat(env.num_envs, 1)

    def __call__(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
        act = _action(env)
        moved = (act.limited - self.prev).abs().mean(dim=-1) / act.cfg.spec.max_step_m
        self.prev = act.limited.clone()
        return moved / env.step_dt

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.prev[env_ids] = torch.tensor(C.NEUTRAL_PAYOUT_M, device=self.prev.device)


def alive_cost(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device) / env.step_dt
