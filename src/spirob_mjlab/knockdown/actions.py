"""Waypoint cable action: one bounded point-to-point move per policy decision.

Mirrors the Lab motor owner (spirob-cable-studio/spirob_cable/rig.py): a new
target is accepted only after both axes hold, and each axis travels to its
target at a fixed commanded speed. The arithmetic for raw -> desired -> limited
is imported from ``contract.py`` so simulation and hardware cannot diverge.

Quantities (all model cable order, metres):
  raw      : policy output (unclipped)
  desired  : neutral + half_span * clip(raw)
  limited  : desired after the per-decision step limit and bounds
  payout   : simulated motor position (what the encoders report)
  ctrl     : MuJoCo tendon-length target = payout + reference_offset
The reference offset models the error between the declared physical rest and
the model neutral; the actor never sees it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import TendonLengthAction, TendonLengthActionCfg

from spirob_mjlab.knockdown import contract as C

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class WaypointCableActionCfg(TendonLengthActionCfg):
    spec: C.ActionSpec = C.DEFAULT.action
    speed_scale_range: tuple[float, float] = (0.85, 1.05)
    """Per-episode multiplier on the commanded speed (motor acceleration, rpm rounding)."""
    reference_offset_range_m: tuple[float, float] = (-0.002, 0.002)
    """Per-episode, per-cable error between encoder payout and model tendon target."""

    def build(self, env: "ManagerBasedRlEnv") -> "WaypointCableAction":
        return WaypointCableAction(self, env)


class WaypointCableAction(TendonLengthAction):
    cfg: WaypointCableActionCfg

    def __init__(self, cfg: WaypointCableActionCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg=cfg, env=env)
        if self.action_dim != 2:
            raise ValueError("Knockdown contract expects exactly two cables")
        spec = cfg.spec
        travel_s = spec.max_step_m / (spec.move_speed_m_s * cfg.speed_scale_range[0])
        if travel_s >= env.step_dt:
            raise ValueError(
                f"Slowest move ({travel_s:.3f} s) must finish inside one decision ({env.step_dt:.3f} s)")
        if abs(env.step_dt - spec.decision_period_s) > 1e-9:
            raise ValueError(
                f"env.step_dt {env.step_dt} must equal contract decision period {spec.decision_period_s}")
        n, dev = self.num_envs, self.device
        self._neutral = torch.tensor(spec.neutral_m, device=dev).repeat(n, 1)
        self._prev_action = torch.zeros(n, 2, device=dev)
        self._desired = self._neutral.clone()
        self._limited = self._neutral.clone()
        self._payout = self._neutral.clone()
        self._speed = torch.full((n, 1), spec.move_speed_m_s, device=dev)
        self._ref_offset = torch.zeros(n, 2, device=dev)
        self._physics_dt = env.physics_dt
        self._processed_actions[:] = self._payout

    # -- exposed for observations / logging --------------------------------
    @property
    def prev_action(self) -> torch.Tensor:
        return self._prev_action

    @property
    def payout(self) -> torch.Tensor:
        return self._payout

    @property
    def desired(self) -> torch.Tensor:
        return self._desired

    @property
    def limited(self) -> torch.Tensor:
        return self._limited

    @property
    def reference_offset(self) -> torch.Tensor:
        return self._ref_offset

    # -- mjlab hooks -------------------------------------------------------
    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        clipped = C.clip_raw(actions, self.cfg.spec)
        self._desired[:] = C.desired_target(clipped, self.cfg.spec)
        self._limited[:] = C.limited_target(self._desired, self._limited, self.cfg.spec)
        self._prev_action[:] = clipped

    def apply_actions(self) -> None:
        step = self._speed * self._physics_dt
        delta = torch.clamp(self._limited - self._payout, min=-step, max=step)
        self._payout += delta
        self._processed_actions[:] = self._payout + self._ref_offset
        super().apply_actions()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids=env_ids)
        self._prev_action[env_ids] = 0.0
        self._desired[env_ids] = self._neutral[env_ids]
        self._limited[env_ids] = self._neutral[env_ids]
        self._payout[env_ids] = self._neutral[env_ids]
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        k = len(ids)
        lo, hi = self.cfg.speed_scale_range
        self._speed[ids, 0] = self.cfg.spec.move_speed_m_s * (lo + (hi - lo) * torch.rand(k, device=self.device))
        olo, ohi = self.cfg.reference_offset_range_m
        self._ref_offset[ids] = olo + (ohi - olo) * torch.rand(k, 2, device=self.device)
        self._processed_actions[ids] = self._payout[ids] + self._ref_offset[ids]
