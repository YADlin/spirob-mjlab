"""Fixed-mean difference waypoints at the selected common speed (contract v3).

Each cable moves half the D change at the same speed, then holds. The existing
Lab motor owner dispatch/settle rules remain authoritative on hardware.
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
class DifferenceWaypointActionCfg(TendonLengthActionCfg):
    spec: C.ActionSpec = C.DEFAULT.action
    reference_offset_range_m: tuple[float, float] = (-0.002, 0.002)
    """Per-episode, per-cable error between encoder payout and model tendon target (unseen by the actor)."""

    def build(self, env: "ManagerBasedRlEnv") -> "DifferenceWaypointAction":
        return DifferenceWaypointAction(self, env)


class DifferenceWaypointAction(TendonLengthAction):
    cfg: DifferenceWaypointActionCfg

    def __init__(self, cfg: DifferenceWaypointActionCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg=cfg, env=env)
        if self._num_targets != 2:
            raise ValueError("Knockdown contract expects two cables")
        s = cfg.spec
        slowest = (s.max_diff_step_m / 2) / s.train_cable_speed_range_m_s[0]
        if slowest >= env.step_dt:
            raise ValueError(f"Largest move at the slowest speed ({slowest:.2f} s) must fit in a decision ({env.step_dt} s)")
        if abs(env.step_dt - s.decision_period_s) > 1e-9:
            raise ValueError(f"env.step_dt {env.step_dt} != contract decision period {s.decision_period_s}")
        n, dev = self.num_envs, self.device
        self._action_dim = 1
        self._raw_actions = torch.zeros(n, 1, device=dev)
        self._prev_action = torch.zeros(n, 1, device=dev)
        self._desired = torch.zeros(n, 1, device=dev)
        self._limited = torch.zeros(n, 1, device=dev)
        self._diff = torch.zeros(n, 1, device=dev)
        self._speed = torch.zeros(n, 1, device=dev)        # per-cable speed for the current move
        self._vmax = torch.full((n, 1), s.cable_speed_max_m_s, device=dev)
        self._ref_offset = torch.zeros(n, 2, device=dev)
        self._physics_dt = env.physics_dt
        self._processed_actions[:] = C.payouts_from_diff(self._diff, s)

    # -- exposed for observations / logging --------------------------------
    @property
    def prev_action(self): return self._prev_action
    @property
    def diff(self): return self._diff
    @property
    def payout(self): return C.payouts_from_diff(self._diff, self.cfg.spec)
    @property
    def limited(self): return self._limited
    @property
    def desired(self): return self._desired
    @property
    def reference_offset(self): return self._ref_offset
    @property
    def vmax(self): return self._vmax

    # -- mjlab hooks -------------------------------------------------------
    def process_actions(self, actions: torch.Tensor) -> None:
        s = self.cfg.spec
        self._raw_actions[:] = actions
        clipped = C.clip_raw(actions, s)
        self._desired[:] = C.desired_diff(clipped, s)
        new = C.limited_diff(self._desired, self._limited, s)
        self._speed[:] = C.cable_speed(new - self._diff, self._vmax, s)
        self._limited[:] = new
        self._prev_action[:] = clipped

    def apply_actions(self) -> None:
        # each cable moves |dD|/2 at self._speed  ->  D changes at 2 * speed
        step = 2 * self._speed * self._physics_dt
        self._diff += torch.clamp(self._limited - self._diff, min=-step, max=step)
        self._processed_actions[:] = C.payouts_from_diff(self._diff, self.cfg.spec) + self._ref_offset
        super().apply_actions()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids=env_ids)
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        k = len(ids)
        s = self.cfg.spec
        start = getattr(self._env, "_knockdown_state", None)
        d0 = start.start_diff[ids] if start is not None else torch.zeros(k, 1, device=self.device)
        self._diff[ids] = d0
        self._limited[ids] = d0
        self._desired[ids] = d0
        self._prev_action[ids] = d0 / s.diff_limit_m      # contract: start memory = measured D / 80 mm
        lo, hi = s.train_cable_speed_range_m_s
        self._vmax[ids, 0] = lo + (hi - lo) * torch.rand(k, device=self.device)
        olo, ohi = self.cfg.reference_offset_range_m
        self._ref_offset[ids] = olo + (ohi - olo) * torch.rand(k, 2, device=self.device)
        self._processed_actions[ids] = C.payouts_from_diff(d0, s) + self._ref_offset[ids]
