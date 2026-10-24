"""Export a trained knockdown checkpoint for the Lab, with a numerical parity check.

Writes into --out (one folder = one deployable policy):
  policy.npz           normalizer + MLP weights (float32) for NumpyPolicy
  policy.onnx          reference ONNX export from mjlab (not used by the Lab)
  contract.json        resolved contract (schema, order, units, action transform, timing)
  contract.py          verbatim copy of the contract source (hash in manifest)
  numpy_policy.py      verbatim copy of the deployment forward pass
  parity_obs.npy       observations recorded from the simulator under the policy
  parity_actions.npy   torch actions for those observations
  manifest.json        hashes, versions, commit, checkpoint, parity result

usage:
  uv run python tools/knockdown/export_policy.py --checkpoint logs/rsl_rl/spirob_knockdown/<run>/model_<i>.pt \
      --out exports/knockdown_<run>_<i> [--device cuda:0] [--parity-envs 16 --parity-steps 24]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import spirob_mjlab  # noqa: F401 (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from spirob_mjlab.knockdown import contract as C
from spirob_mjlab.knockdown import numpy_policy as NP
from spirob_mjlab.knockdown.entities import lab_model_sha256

TASK = "Mjlab-SpiRob-EggKnockdown"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, cwd=Path(__file__).parents[2]).strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable: {exc}"


def actor_module(runner) -> torch.nn.Module:
    alg = runner.alg
    for name in ("actor", "policy"):
        m = getattr(alg, name, None)
        if isinstance(m, torch.nn.Module) and hasattr(m, "mlp"):
            return m
    raise RuntimeError("Could not locate the RSL-RL actor module (expected alg.actor with .mlp)")


def extract(actor) -> dict:
    norm = actor.obs_normalizer
    if isinstance(norm, torch.nn.Identity):
        mean, std, eps = np.zeros(actor.obs_dim), np.ones(actor.obs_dim), 0.0
    else:
        mean, std, eps = norm.mean.cpu().numpy(), norm.std.cpu().numpy(), float(norm.eps)
    linears, acts = [], set()
    for layer in actor.mlp:
        if isinstance(layer, torch.nn.Linear):
            linears.append(layer)
        elif isinstance(layer, torch.nn.ELU):
            acts.add("elu")
        elif isinstance(layer, torch.nn.Tanh):
            acts.add("tanh")
        elif isinstance(layer, torch.nn.ReLU):
            acts.add("relu")
        else:
            raise RuntimeError(f"Unsupported layer in actor MLP: {layer}")
    if len(acts) != 1:
        raise RuntimeError(f"Expected one hidden activation type, found {acts}")
    arrays = {"obs_mean": mean.astype(np.float32), "obs_std": std.astype(np.float32), "obs_eps": np.float32(eps)}
    for i, lin in enumerate(linears):
        arrays[f"W{i}"] = lin.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"b{i}"] = lin.bias.detach().cpu().numpy().astype(np.float32)
    meta = {"n_layers": len(linears), "activation": acts.pop(), "obs_dim": int(mean.shape[0]),
            "act_dim": int(linears[-1].out_features)}
    arrays["meta_json"] = np.array(json.dumps(meta))
    return arrays


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--parity-envs", type=int, default=16)
    p.add_argument("--parity-steps", type=int, default=24)
    p.add_argument("--task", default=TASK)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    ckpt = Path(a.checkpoint).resolve()

    env_cfg = load_env_cfg(a.task)
    env_cfg.scene.num_envs = a.parity_envs
    agent_cfg = load_rl_cfg(a.task)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device=a.device), clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=a.device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=a.device)
    policy = runner.get_inference_policy(device=a.device)

    # Export weights.
    arrays = extract(actor_module(runner))
    np.savez(out / "policy.npz", **arrays)
    try:
        runner.export_policy_to_onnx(str(out), "policy.onnx")
    except Exception as exc:  # ONNX is reference-only
        (out / "policy.onnx.FAILED.txt").write_text(str(exc))

    # Parity on observations recorded from simulator rollouts (includes resets).
    obs_log, act_log = [], []
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(a.parity_steps):
            act = policy(obs)
            obs_log.append(obs["actor"].cpu().numpy().copy())
            act_log.append(act.cpu().numpy().copy())
            obs, _, _, _ = env.step(act)
    O = np.concatenate(obs_log).astype(np.float64)
    A = np.concatenate(act_log).astype(np.float64)
    np.save(out / "parity_obs.npy", O)
    np.save(out / "parity_actions.npy", A)
    shutil.copy2(Path(C.__file__), out / "contract.py")
    shutil.copy2(Path(NP.__file__), out / "numpy_policy.py")
    npp = NP.NumpyPolicy(out / "policy.npz")
    err = float(np.max(np.abs(npp(O) - A)))
    passed = err < 1e-4
    contract = C.write_contract_json(out / "contract.json")

    manifest = {
        "task": a.task,
        "checkpoint": str(ckpt), "checkpoint_sha256": sha(ckpt),
        "policy_npz_sha256": sha(out / "policy.npz"),
        "contract_id": C.CONTRACT_ID, "contract_source_sha256": contract["contract_source_sha256"],
        "numpy_policy_sha256": sha(out / "numpy_policy.py"),
        "lab_model_sha256": lab_model_sha256(),
        "obs_dim": npp.obs_dim, "act_dim": npp.act_dim,
        "parity": {"n_obs": int(O.shape[0]), "max_abs_error": err, "threshold": 1e-4, "passed": passed},
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "versions": {k: md.version(k) for k in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "rsl-rl-lib", "numpy")
                     if _has(k)},
        "python": platform.python_version(), "device": a.device,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest["parity"]))
    if not passed:
        raise SystemExit("PARITY FAILED: do not deploy this export")


def _has(dist: str) -> bool:
    try:
        md.version(dist)
        return True
    except md.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
