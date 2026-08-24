"""Custom action terms for SpiRob mjlab tasks.

This module adds a rate-limited tendon-length action. The policy may still ask
for the full valid tendon-length range, but the command sent to MuJoCo is ramped
by at most `max_delta_per_step` at each policy/control step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import TendonLengthAction, TendonLengthActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class RateLimitedTendonLengthActionCfg(TendonLengthActionCfg):
    """Tendon-length action with command-rate limiting.

    The inherited TendonLengthActionCfg first converts raw policy actions into
    desired tendon-length targets using:

        desired = raw_action * scale + offset

    and then applies any configured clip range. This subclass preserves that
    full target range, but limits how far the actually commanded target may move
    per policy/control step.
    """

    max_delta_per_step: float = 1.5e-4
    """Maximum tendon-length command change per policy/control step, in metres.

    Example: if the effective control timestep is 0.01 s, then 1.5e-4 m/step is
    0.015 m/s. A target jump from 0.22 to 0.25 m therefore takes about 2 s.
    """

    def build(self, env: ManagerBasedRlEnv) -> "RateLimitedTendonLengthAction":
        return RateLimitedTendonLengthAction(self, env)


class RateLimitedTendonLengthAction(TendonLengthAction):
    """Tendon-length action that ramps toward the processed target."""

    cfg: RateLimitedTendonLengthActionCfg

    def __init__(self, cfg: RateLimitedTendonLengthActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        self._desired_actions = torch.zeros_like(self._processed_actions)
        self._rate_limited_actions = self._initial_target_tensor()
        self._processed_actions[:] = self._rate_limited_actions

    def _initial_target_tensor(self) -> torch.Tensor:
        """Initial commanded tendon target, normally the configured offset/rest."""
        if isinstance(self._offset, torch.Tensor):
            target = self._offset.clone()
        else:
            target = torch.full_like(self._processed_actions, float(self._offset))

        if self.cfg.clip is not None:
            target = torch.clamp(
                target,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
        return target

    @property
    def desired_action(self) -> torch.Tensor:
        """Desired tendon target after scale/offset/clip but before rate limiting."""
        return self._desired_actions

    @property
    def rate_limited_action(self) -> torch.Tensor:
        """Commanded tendon target after rate limiting."""
        return self._rate_limited_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Convert policy action to full-range target, then apply rate limit."""
        super().process_actions(actions)

        self._desired_actions[:] = self._processed_actions
        delta = self._desired_actions - self._rate_limited_actions
        delta = torch.clamp(
            delta,
            min=-float(self.cfg.max_delta_per_step),
            max=float(self.cfg.max_delta_per_step),
        )
        self._rate_limited_actions[:] = self._rate_limited_actions + delta

        if self.cfg.clip is not None:
            self._rate_limited_actions[:] = torch.clamp(
                self._rate_limited_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

        # Parent TendonLengthAction.apply_actions writes self._processed_actions
        # to the tendon actuator target. Replace it with the ramped command.
        self._processed_actions[:] = self._rate_limited_actions

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset command memory to the tendon rest/offset target."""
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids=env_ids)
        initial = self._initial_target_tensor()
        self._desired_actions[env_ids] = initial[env_ids]
        self._rate_limited_actions[env_ids] = initial[env_ids]
        self._processed_actions[env_ids] = initial[env_ids]
