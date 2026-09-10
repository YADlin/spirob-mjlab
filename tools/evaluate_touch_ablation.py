"""Nominal-only deterministic checkpoint evaluation for TCH-fixed-v1.

Reuses the existing Stage-1 episode recorder with auto_reset=False. No grid,
sector, spawn perturbation, domain randomization or policy sampling is used.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_stage1 as s1
from spirob_mjlab.touch_ablation import TASK_IDS, PROTOCOL, ablated_touch


def evaluate(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    checkpoint = Path(args.checkpoint_file).resolve()
    digest = s1._sha256_file(checkpoint)
    if digest != args.expected_sha256:
        raise ValueError("Checkpoint hash mismatch")
    commit = s1._run_git("rev-parse", "HEAD")
    if s1._run_git("status", "--porcelain"):
        raise RuntimeError("Commit changes before scientific evaluation")
    if commit != args.training_git_commit:
        raise ValueError("Training and evaluation must use the same commit")
    output.mkdir(parents=True)
    task_id = TASK_IDS[args.arm]
    cfg = s1.load_env_cfg(task_id, play=False)
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    cfg.seed = 3210
    s1.configure_torch_backends(allow_tf32=True, deterministic=True)
    s1.seed_rng(3210, torch_deterministic=True)
    env = s1.ManagerBasedRlEnv(cfg=cfg, device=args.device)
    agent = s1.load_rl_cfg(task_id)
    vec = s1.RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    rows = []
    identity = {
        "protocol_version": PROTOCOL,
        "run_id": args.run_id,
        "training_seed": args.training_seed,
        "training_git_commit": commit,
        "evaluator_git_commit": commit,
        "evaluator_git_dirty": False,
        "task_package_sha256": s1._sha256_tree(s1._repo_root() / "src/spirob_mjlab"),
        "dependency_lock_sha256": s1._sha256_file(s1._repo_root() / "uv.lock"),
        "checkpoint_name": checkpoint.name,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": digest,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "policy_mode": "deterministic",
    }
    try:
        runner = s1.MjlabOnPolicyRunner(vec, asdict(agent), device=args.device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True,
                    map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        with (output / "trace_first_episode.csv").open("w", newline="") as f:
            trace = csv.writer(f)
            trace.writerow(["step", "time_s", "raw_active_count", "visible_active_count",
                            "egg_x_m", "egg_y_m", "egg_z_m", "action_0", "action_1"])
            for repeat in range(args.repeats):
                counts = np.zeros(42, dtype=np.int64)
                nonfinite = 0
                steps = 0

                def audited_policy(obs):
                    nonlocal nonfinite, steps
                    raw = env.sim.data.sensordata[:, :42]
                    bits = (raw > 1e-4).float()
                    visible = ablated_touch(env, enabled=args.arm == "touch")
                    # Audit the actual full actor AND critic tensors, not just
                    # the standalone masking function.
                    for group in ("actor", "critic"):
                        if obs[group].shape[-1] != 93:
                            raise RuntimeError("Unexpected observation dimension")
                        torch.testing.assert_close(obs[group][:, -42:], visible)
                    nonfinite += int((~torch.isfinite(raw)).sum().item())
                    counts[:] += bits[0].detach().cpu().numpy().astype(np.int64)
                    action = policy(obs)
                    if repeat == 0:
                        egg = s1._snapshot(env)["egg"]
                        trace.writerow([steps, steps * env.step_dt, int(bits.sum()),
                                        int(visible.sum()), *egg,
                                        *action[0].detach().cpu().numpy().tolist()])
                    steps += 1
                    return action

                episode_cfg = s1.Stage1EvaluateConfig(
                    run_id=args.run_id, training_seed=args.training_seed,
                    training_git_commit=commit, evaluation_seed=3210 + repeat,
                    protocol="nominal", policy_mode="deterministic",
                )
                row = s1._run_episode(vec, audited_policy, episode_cfg,
                                      s1.Condition("nominal_repeat", repeat, 0, 0),
                                      {**identity, "evaluation_seed": 3210 + repeat}, repeat, "")
                # Unknown/overlapping outcomes and NaNs cannot contribute successes.
                row["success"] = row["outcome"] == "success"
                row.update(arm=args.arm, checkpoint_index=args.checkpoint_index,
                           completed_updates=args.checkpoint_index + 1,
                           training_transitions=(args.checkpoint_index + 1) * 256 * 32,
                           raw_sensor_nonfinite_count=nonfinite,
                           raw_active_sensor_steps=int(counts.sum()),
                           observed_steps=steps,
                           raw_sensor_active_counts=json.dumps(counts.tolist()))
                rows.append(row)
                print(f"[TCH] {args.arm} {checkpoint.name} {repeat + 1}/{args.repeats}: {row['outcome']}", flush=True)
    finally:
        vec.close()
    with (output / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        **identity, "arm": args.arm, "checkpoint_index": args.checkpoint_index,
        "repeats": len(rows), "success_fraction": np.mean([r["success"] for r in rows]),
        "interpretation": "Execution repeats at one exact spawn; training seed is the independent unit.",
        "device": args.device, "packages": {
            name: s1.importlib.metadata.version(name) for name in
            ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib", "torch")
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    s1._write_manifest(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=TASK_IDS, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--training-seed", type=int, required=True)
    p.add_argument("--training-git-commit", required=True)
    p.add_argument("--checkpoint-file", required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--checkpoint-index", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.repeats < 1:
        p.error("repeats must be positive")
    if Path(args.checkpoint_file).name != f"model_{args.checkpoint_index}.pt":
        p.error("checkpoint index does not match filename")
    evaluate(args)


if __name__ == "__main__":
    main()
