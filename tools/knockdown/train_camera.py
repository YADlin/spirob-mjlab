"""Fine-tune the existing actor/critic on rendered RGB and common-speed waypoints.

Starts a new run and optimizer. The source checkpoint is never overwritten.
Use --smoke for a two-step, one-update pipeline check, not a trained result.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import subprocess

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from spirob_mjlab.knockdown import contract as C
from spirob_mjlab.knockdown.entities import SOURCE_XML, sha256
from spirob_mjlab.knockdown.env_cfg import knockdown_env_cfg
from spirob_mjlab.knockdown.rl_cfg import knockdown_ppo_runner_cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.envs < 1 or a.iterations < 1:
        p.error("envs and iterations must be positive")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    cfg = knockdown_env_cfg(rendered_camera=True, fixed=a.smoke, randomize=not a.smoke)
    cfg.scene.num_envs = 2 if a.smoke else a.envs
    cfg.seed = a.seed
    agent = knockdown_ppo_runner_cfg()
    agent.seed, agent.logger = a.seed, "tensorboard"
    agent.algorithm.learning_rate = 1e-4
    agent.algorithm.schedule = "fixed"
    agent.run_name = "kd-camera-v3-smoke" if a.smoke else "kd-camera-v3"
    if a.smoke:
        agent.num_steps_per_env = 2
        agent.algorithm.num_mini_batches = 1
        agent.algorithm.num_learning_epochs = 1
    manifest = dict(
        contract_id=C.CONTRACT_ID, contract_sha256=C.source_file_sha256(),
        source_checkpoint=str(Path(a.checkpoint).resolve()), source_checkpoint_sha256=sha256(Path(a.checkpoint)),
        source_model_sha256=sha256(SOURCE_XML), seed=a.seed, smoke=a.smoke,
        envs=cfg.scene.num_envs, iterations=1 if a.smoke else a.iterations,
        perception="MuJoCo-Warp RGB -> shared Lab black-egg detector", sensor_resolution=[640, 360],
        common_speed_range_mm_s=[5, 20], device=a.device,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        versions={k: importlib.metadata.version(k) for k in
                  ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "rsl-rl-lib", "opencv-python-headless")},
        agent=asdict(agent),
    )
    (out / "camera_run.json").write_text(json.dumps(manifest, indent=2))
    base = ManagerBasedRlEnv(cfg, device=a.device)
    env = RslRlVecEnvWrapper(base, clip_actions=agent.clip_actions)
    try:
        runner = MjlabOnPolicyRunner(env, asdict(agent), str(out), device=a.device)
        # Use the normalizer + both networks, not the previous optimizer/iteration.
        saved = torch.load(a.checkpoint, map_location=a.device, weights_only=True)
        runner.alg.load(saved, {"actor": True, "critic": True}, strict=True)
        runner.learn(num_learning_iterations=manifest["iterations"], init_at_random_ep_len=False)
    finally:
        env.close()


if __name__ == "__main__":
    main()
