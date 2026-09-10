"""TCH-v1: fixed-scene ablation of binary touch information.

Both arms retain the XML sensors, 93 observation entries and the same PPO
architecture. Only the last 42 actor AND critic entries change. No Stage-2
configuration or checkpoint is used.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.rl import MjlabOnPolicyRunner

from spirob_mjlab import mdp
from spirob_mjlab.env_cfgs import spirob_egg_to_bucket_stage1_env_cfg
from spirob_mjlab.rl_cfg import spirob_ppo_runner_cfg

TASK_IDS = {
    "touch": "Mjlab-SpiRob-EggToBucket-TouchAblation-Fixed-Touch",
    "no-touch": "Mjlab-SpiRob-EggToBucket-TouchAblation-Fixed-NoTouch",
}
PROTOCOL = "TCH-fixed-v1"
SEEDS = (42, 43, 44, 45, 46, 47)
CHECKPOINT_INDICES = (0, 100, 200, 300, 400, 499)


def ablated_touch(env, enabled: bool = True):
    values = mdp.touch_values(env)
    return values if enabled else torch.zeros_like(values)


def raw_touch_active_count(env):
    """Diagnostic only: not available to the actor, critic or reward."""
    return mdp.touch_values(env).sum(dim=-1)


def touch_ablation_env_cfg(*, enabled: bool, play: bool = False):
    cfg = spirob_egg_to_bucket_stage1_env_cfg(play=play)
    cfg.episode_length_s = 8.0
    for group in ("actor", "critic"):
        cfg.observations[group] = deepcopy(cfg.observations[group])
        term = cfg.observations[group].terms["touch"]
        term.func = ablated_touch
        term.params = {"enabled": enabled}
    cfg.metrics = dict(cfg.metrics)
    cfg.metrics["raw_touch_active_count"] = MetricsTermCfg(
        func=raw_touch_active_count, reduce="mean"
    )
    return cfg


def touch_ablation_ppo_cfg():
    cfg = spirob_ppo_runner_cfg(stage2=False)
    cfg.experiment_name = "spirob_touch_ablation"
    cfg.max_iterations = 500
    cfg.save_interval = 50
    cfg.resume = False
    return cfg


def model_state_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class TouchAblationRunner(MjlabOnPolicyRunner):
    """Standard learning; only adds a pre-learning state audit."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        if log_dir is not None:
            audit = {
                "protocol": PROTOCOL,
                "actor_initial_sha256": model_state_hash(self.alg.actor),
                "critic_initial_sha256": model_state_hash(self.alg.critic),
                "actor_parameters": sum(p.numel() for p in self.alg.actor.parameters()),
                "critic_parameters": sum(p.numel() for p in self.alg.critic.parameters()),
            }
            path = Path(log_dir) / "touch_initial_state.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(audit, indent=2) + "\n")
