"""Load the declared seven-input ELU actor without creating a physics environment.

Strict RSL-RL loading retains its normalizer and rejects incompatible tensors.
This does not certify a checkpoint for deployment under a changed contract.
"""
from dataclasses import asdict

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from spirob_mjlab.knockdown.rl_cfg import knockdown_ppo_runner_cfg


def load_actor(path, device="cpu"):
    cfg = asdict(knockdown_ppo_runner_cfg().actor)
    actor = MLPModel(
        TensorDict({"actor": torch.zeros(1, 7)}, batch_size=[1]),
        {"actor": ["actor"]}, "actor", 1,
        hidden_dims=cfg["hidden_dims"], activation=cfg["activation"],
        obs_normalization=cfg["obs_normalization"], distribution_cfg=cfg["distribution_cfg"],
    ).to(device)
    data = torch.load(path, map_location=device, weights_only=True)
    actor.load_state_dict(data["actor_state_dict"], strict=True)
    actor.eval()
    return actor


def action(actor, obs):
    with torch.inference_mode():
        x = torch.as_tensor(obs, dtype=torch.float32, device=next(actor.parameters()).device)
        if x.ndim == 1:
            x = x[None]
        return actor(TensorDict({"actor": x}, batch_size=x.shape[:-1])).cpu().numpy()
