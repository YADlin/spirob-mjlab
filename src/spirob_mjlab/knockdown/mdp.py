"""MDP terms for the SpiRob egg-knockdown task (contract v2).

Actor information is limited to the camera/encoder contract. True egg state, robot keypoints,
contact, visibility truth and randomised parameters are privileged (rewards, terminations, critic).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from spirob_mjlab.knockdown import contract as C

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

ACTION_TERM = "cable_diff"
CONTACT_SENSOR = "egg_robot_contact"
EGG_SETTLED_Z = 0.0750           # egg root on the knockdown pedestal (CPU MuJoCo 3.8.1: 74.98 mm)
PEDESTAL_ROOT_Z = 0.002
POSE_TABLE = Path(__file__).resolve().parents[1] / "assets" / "knockdown" / "pose_table.json"


# ---------------------------------------------------------------------------
# Shared helpers and per-episode state
# ---------------------------------------------------------------------------
def _action(env):
    return env.action_manager.get_term(ACTION_TERM)


def _c_origin_w(env) -> torch.Tensor:
    return env.scene.env_origins[:, :2] + torch.tensor(C.C_ORIGIN_MODEL_XY, device=env.device)


def world_xy_to_c(env, xy_w: torch.Tensor) -> torch.Tensor:
    origin = _c_origin_w(env)
    shape = (-1,) + (1,) * (xy_w.dim() - 2) + (2,)
    rel = xy_w - origin.view(shape)
    return torch.stack((rel[..., 0], -rel[..., 1]), dim=-1)


def egg_pos_w(env) -> torch.Tensor:
    return env.scene["egg"].data.root_link_pos_w


def egg_uv_true(env) -> torch.Tensor:
    return world_xy_to_c(env, egg_pos_w(env)[:, :2])


class PoseTable:
    def __init__(self, device):
        t = json.loads(POSE_TABLE.read_text())
        rows = t["rows"]
        self.diff = torch.tensor([r["diff_mm"] / 1000 for r in rows], device=device)
        self.qpos = torch.tensor([r["qpos"] for r in rows], device=device)
        self.link_uv = torch.tensor([r["link_uv"] for r in rows], device=device)
        self.halfwidth = torch.tensor(t["link_halfwidth_m"], device=device)
        self.zero = int(torch.argmin(self.diff.abs()))

    def clearance(self, idx: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """idx (n,), uv (n, k, 2) -> (n, k) distance to the robot outline in that shape."""
        pts = self.link_uv[idx]                                  # (n, 21, 2)
        dist = torch.linalg.norm(uv[:, :, None, :] - pts[:, None, :, :], dim=-1) - self.halfwidth
        return dist.min(dim=-1).values


class KnockdownState:
    def __init__(self, env):
        n, dev = env.num_envs, env.device
        self.cam_bias = torch.zeros(n, 2, device=dev)
        self.egg_uv_start_obs = torch.zeros(n, 2, device=dev)
        self.ped_uv = torch.zeros(n, 2, device=dev)
        self.last_seen = torch.zeros(n, 2, device=dev)
        self.visible = torch.ones(n, 1, device=dev)
        self.start_diff = torch.zeros(n, 1, device=dev)
        self.robot_dz = torch.zeros(n, 1, device=dev)
        self.table = PoseTable(dev)


def state(env) -> KnockdownState:
    s = getattr(env, "_knockdown_state", None)
    if s is None:
        s = KnockdownState(env)
        env._knockdown_state = s
    return s


# ---------------------------------------------------------------------------
# Camera model (sim stand-in for lab/rl/egg_vision.py)
# ---------------------------------------------------------------------------
def in_camera_view(uv: torch.Tensor, cam: C.CameraSpec = C.DEFAULT.camera) -> torch.Tensor:
    cu, cv = cam.centre_uv_m
    hx, hy = cam.visible_half_extent_uv_m
    return ((uv[:, 0] - cu).abs() <= hx) & ((uv[:, 1] - cv).abs() <= hy)


def camera_egg(env, cam: C.CameraSpec = C.DEFAULT.camera):
    """(egg_uv_obs, visible) with per-episode bias, per-frame noise, last-seen hold outside the view."""
    s = state(env)
    true = egg_uv_true(env)
    vis = in_camera_view(true, cam)
    seen = true + s.cam_bias + cam.egg_noise_m * torch.randn_like(true)
    s.last_seen = torch.where(vis[:, None], seen, s.last_seen)
    s.visible = vis.float().unsqueeze(-1)
    return s.last_seen.clone(), s.visible.clone()


# ---------------------------------------------------------------------------
# Reset: robot start shape, robot height, egg + pedestal placement
# ---------------------------------------------------------------------------
def sample_region_uv(shape, device, abs_u_range, v_range, side="both") -> torch.Tensor:
    a0, a1 = abs_u_range
    v0, v1 = v_range
    au = a0 + (a1 - a0) * torch.rand(shape, device=device)
    v = v0 + (v1 - v0) * torch.rand(shape, device=device)
    if side == "both":
        sign = torch.where(torch.rand(shape, device=device) < 0.5, -1.0, 1.0)
    else:
        sign = torch.full(shape, 1.0 if side == "positive" else -1.0, device=device)
    return torch.stack((sign * au, v), dim=-1)


def reset_knockdown_episode(
    env, env_ids: torch.Tensor | None,
    abs_u_range: tuple[float, float], v_range: tuple[float, float], side: str = "both",
    start_shape_prob: float = 0.7, fixed_uv: tuple[float, float] | None = None,
    robot_dz_range: tuple[float, float] = (-0.015, 0.0), clearance_m: float = C.DEFAULT.autostart.clearance_m,
    candidates: int = 32, yaw_range_rad: float = math.pi, cam: C.CameraSpec = C.DEFAULT.camera,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    if n == 0:
        return
    dev = env.device
    s = state(env)
    tab = s.table

    # 1. start shape: any tabulated difference (the rig continues from where it stopped), or straight
    idx = torch.randint(0, len(tab.diff), (n,), device=dev)
    idx = torch.where(torch.rand(n, device=dev) < start_shape_prob, idx, torch.full_like(idx, tab.zero))
    ov_d = getattr(env, "_kd_diff_override", None)
    if ov_d is not None:
        want = ov_d[env_ids]
        use = torch.isfinite(want)
        nearest = torch.argmin((tab.diff[None, :] - want[:, None].nan_to_num(0)).abs(), dim=1)
        idx = torch.where(use, nearest, idx)

    # 2. egg placement clear of the robot in that shape
    if fixed_uv is not None:
        uv = torch.tensor(fixed_uv, device=dev).repeat(n, 1)
        idx = torch.where(tab.clearance(idx, uv[:, None, :])[:, 0] > clearance_m, idx, torch.full_like(idx, tab.zero))
    else:
        cand = sample_region_uv((n, candidates), dev, abs_u_range, v_range, side)
        ok = tab.clearance(idx, cand) > clearance_m
        first = torch.argmax(ok.int(), dim=1)
        uv = cand[torch.arange(n, device=dev), first]
        none = ~ok.any(dim=1)
        if none.any():  # fall back to the straight robot (always valid for the region)
            idx = torch.where(none, torch.full_like(idx, tab.zero), idx)
            ok0 = tab.clearance(idx, cand) > clearance_m
            uv = torch.where(none[:, None], cand[torch.arange(n, device=dev), torch.argmax(ok0.int(), dim=1)], uv)
    ov = getattr(env, "_kd_uv_override", None)
    if ov is not None:
        o = ov[env_ids]
        use = torch.isfinite(o).all(dim=-1)
        uv = torch.where(use[:, None], o, uv)

    # 3. robot: joint state of the start shape, base height offset (real robot droops 12-20 mm)
    robot: Entity = env.scene["robot"]
    robot.write_joint_state_to_sim(tab.qpos[idx], torch.zeros_like(tab.qpos[idx]), env_ids=env_ids)
    dz = robot_dz_range[0] + (robot_dz_range[1] - robot_dz_range[0]) * torch.rand(n, device=dev)
    root = robot.data.default_root_state[env_ids].clone()
    pose = torch.cat((root[:, 0:3] + env.scene.env_origins[env_ids], root[:, 3:7]), dim=-1)
    pose[:, 2] += dz
    robot.write_mocap_pose_to_sim(pose, env_ids=env_ids)
    s.start_diff[env_ids] = tab.diff[idx].unsqueeze(-1)
    s.robot_dz[env_ids] = dz.unsqueeze(-1)

    # 4. pedestal + egg together
    origin = _c_origin_w(env)[env_ids]
    xy = torch.stack((origin[:, 0] + uv[:, 0], origin[:, 1] - uv[:, 1]), dim=-1)
    ped: Entity = env.scene["pedestal"]
    ppose = torch.zeros(n, 7, device=dev)
    ppose[:, 0:2] = xy
    ppose[:, 2] = env.scene.env_origins[env_ids, 2] + PEDESTAL_ROOT_Z
    ppose[:, 3] = 1.0
    ped.write_mocap_pose_to_sim(ppose, env_ids=env_ids)
    egg: Entity = env.scene["egg"]
    yaw = yaw_range_rad * (2 * torch.rand(n, device=dev) - 1)
    r = torch.zeros(n, 13, device=dev)
    r[:, 0:2] = xy
    r[:, 2] = env.scene.env_origins[env_ids, 2] + EGG_SETTLED_Z
    r[:, 3] = torch.cos(yaw / 2)
    r[:, 6] = torch.sin(yaw / 2)
    egg.write_root_state_to_sim(r, env_ids=env_ids)

    # 5. camera estimator state
    s.ped_uv[env_ids] = uv
    s.cam_bias[env_ids] = cam.egg_bias_m * (2 * torch.rand(n, 2, device=dev) - 1)
    start = uv + s.cam_bias[env_ids] + cam.egg_noise_m * torch.randn(n, 2, device=dev)
    s.egg_uv_start_obs[env_ids] = start
    s.last_seen[env_ids] = start
    s.visible[env_ids] = 1.0


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
def actor_contract_obs(env) -> torch.Tensor:
    act = _action(env)
    s = state(env)
    egg_uv, vis = camera_egg(env)
    return C.build_actor_obs(egg_uv, vis, s.egg_uv_start_obs, act.diff, act.prev_action)


def critic_privileged(env, keypoint_cfg: SceneEntityCfg) -> torch.Tensor:
    act = _action(env)
    s = state(env)
    robot: Entity = env.scene["robot"]
    egg_w = egg_pos_w(env)
    egg_uv = world_xy_to_c(env, egg_w[:, :2])
    vis_true = in_camera_view(egg_uv).float().unsqueeze(-1)
    egg_rel = torch.cat((egg_uv - s.ped_uv, egg_w[:, 2:3] - env.scene.env_origins[:, 2:3] - EGG_SETTLED_Z), dim=-1)
    egg_vel = env.scene["egg"].data.root_link_lin_vel_w
    kp = world_xy_to_c(env, robot.data.body_link_pos_w[:, keypoint_cfg.body_ids, :2])
    ten = robot.data.tendon_len[:, :2]
    touched = egg_touched(env).float().unsqueeze(-1)
    clean = C.build_actor_obs(egg_uv, vis_true, s.ped_uv, act.diff, act.prev_action)
    frac = (env.episode_length_buf.float() / env.max_episode_length).unsqueeze(-1)
    return torch.cat((clean, egg_rel / 0.05, egg_vel.clamp(-2, 2), kp.flatten(1) / 0.1,
                      (ten - C.MEAN_PAYOUT_M) / 0.06, act.reference_offset / 0.002, act.vmax / 0.015,
                      s.robot_dz / 0.015, touched, frac), dim=-1)


# ---------------------------------------------------------------------------
# Classification (privileged)
# ---------------------------------------------------------------------------
def egg_touched(env) -> torch.Tensor:
    d = env.scene[CONTACT_SENSOR].data
    return ((d.current_contact_time > 0) | (d.last_contact_time > 0)).any(dim=-1)


def egg_below_support(env, spec: C.SuccessSpec = C.DEFAULT.success) -> torch.Tensor:
    z = egg_pos_w(env)[:, 2] - env.scene.env_origins[:, 2]
    return z < (spec.sim_support_top_z_m - spec.sim_fall_margin_m)


def numerical_fault(env) -> torch.Tensor:
    return ~torch.isfinite(env.sim.data.qacc).all(dim=-1) | ~torch.isfinite(egg_pos_w(env)).all(dim=-1)


# precedence: numerical_fault > knocked_down > spontaneous_fall > egg_out_of_bounds > time_out
def term_numerical_fault(env):
    return numerical_fault(env)


def term_knocked_down(env):
    return ~numerical_fault(env) & egg_below_support(env) & egg_touched(env)


def term_spontaneous_fall(env):
    return ~numerical_fault(env) & egg_below_support(env) & ~egg_touched(env)


def term_egg_out_of_bounds(env, xy_limit: float = 0.40):
    rel = egg_pos_w(env)[:, :2] - env.scene.env_origins[:, :2]
    return ~numerical_fault(env) & ~egg_below_support(env) & (rel.abs() > xy_limit).any(dim=-1)


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
def knockdown_bonus(env):
    return term_knocked_down(env).float() / env.step_dt


class _PotentialDelta:
    """Reward = Phi(s_t) - Phi(s_{t-1}); telescopes, so it cannot be farmed."""

    def __init__(self, cfg, env):
        self.prev = torch.zeros(env.num_envs, device=env.device)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def potential(self, env, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def __call__(self, env, **kwargs):
        phi = self.potential(env, **kwargs)
        out = torch.where(self.fresh, torch.zeros_like(phi), phi - self.prev)
        self.prev = phi.detach().clone()
        self.fresh[:] = False
        return out / env.step_dt

    def reset(self, env_ids=None):
        self.fresh[slice(None) if env_ids is None else env_ids] = True


class approach_progress(_PotentialDelta):
    def potential(self, env, keypoint_cfg: SceneEntityCfg, d_max: float = 0.10):
        kp = env.scene["robot"].data.body_link_pos_w[:, keypoint_cfg.body_ids, :2]
        d = torch.linalg.norm(kp - egg_pos_w(env)[:, None, :2], dim=-1).min(dim=-1).values
        return -d.clamp(0.0, d_max) / d_max


class egg_displacement_progress(_PotentialDelta):
    def potential(self, env, d_max: float = 0.03):
        d = torch.linalg.norm(egg_uv_true(env) - state(env).ped_uv, dim=-1)
        return d.clamp(0.0, d_max) / d_max


class command_motion_cost:
    """|change of the commanded difference| / max step (1.0 for a full 8 mm step)."""

    def __init__(self, cfg, env):
        self.prev = torch.zeros(env.num_envs, 1, device=env.device)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env):
        act = _action(env)
        moved = ((act.limited - self.prev).abs() / act.cfg.spec.max_diff_step_m).squeeze(-1)
        moved = torch.where(self.fresh, torch.zeros_like(moved), moved)
        self.prev = act.limited.clone()
        self.fresh[:] = False
        return moved / env.step_dt

    def reset(self, env_ids=None):
        self.fresh[slice(None) if env_ids is None else env_ids] = True


def alive_cost(env):
    return torch.ones(env.num_envs, device=env.device) / env.step_dt
