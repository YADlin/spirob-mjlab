"""Frozen-policy evaluation of the knockdown task on held-out placements.

Each environment runs exactly one episode per assigned placement; only the first
termination of each episode is counted. All outcomes are retained.

Placements (default 200): a 30-point boundary/interior grid of the evaluation
rectangle on both sides, plus 170 uniform samples from it drawn with a held-out
seed. Training never uses --placement-seed values >= 10000.

Outcomes
  knocked_down       success (actual egg-floor contact after robot contact)
  time_out           task failure
  egg_out_of_bounds  task failure
  spontaneous_fall   invalid episode (egg fell without robot contact) - reported separately
  numerical_fault    infrastructure fault - reported separately

usage:
  uv run python tools/knockdown/evaluate.py --checkpoint <model.pt> --out eval/<name> [--episodes 200]
  uv run python tools/knockdown/evaluate.py --agent hold --out eval/hold_baseline     # no-motion check; zero requests D=0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import spirob_mjlab  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from spirob_mjlab.knockdown import mdp
from spirob_mjlab.knockdown.camera import camera_manifest
from spirob_mjlab.knockdown.env_cfg import EVAL_ABS_U, EVAL_V
from spirob_mjlab.knockdown.entities import SOURCE_XML, sha256

OUTCOMES = ("numerical_fault", "knocked_down", "spontaneous_fall", "egg_out_of_bounds", "time_out")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def placements(n: int, seed: int) -> np.ndarray:
    grid = [(s * au, v) for s in (-1, 1) for au in (EVAL_ABS_U[0], sum(EVAL_ABS_U) / 2, EVAL_ABS_U[1])
            for v in np.linspace(EVAL_V[0], EVAL_V[1], 5)]
    rng = np.random.default_rng(seed)
    k = max(0, n - len(grid))
    rand = np.stack([rng.choice([-1.0, 1.0], k) * rng.uniform(*EVAL_ABS_U, k), rng.uniform(*EVAL_V, k)], axis=1)
    pts = np.concatenate([np.array(grid), rand])[:n]
    return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    p.add_argument("--agent", choices=["trained", "hold", "zero"], default="trained")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--envs", type=int, default=100)
    p.add_argument("--placement-seed", type=int, default=10001)
    p.add_argument("--sim-seed", type=int, default=20001)
    p.add_argument("--cable-speed-mm-s", type=float, help="Fixed common speed in [5, 20]; default samples the trained range")
    p.add_argument("--nominal", action="store_true", help="disable dynamics randomization")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--task", default="Mjlab-SpiRob-EggKnockdown-Eval")
    a = p.parse_args()
    if a.agent == "trained" and not a.checkpoint:
        p.error("--checkpoint is required for --agent trained")
    if a.episodes < 1 or a.envs < 1:
        p.error("episodes and envs must be positive")
    if a.cable_speed_mm_s is not None and not 5 <= a.cable_speed_mm_s <= 20:
        p.error("common speed must be in [5, 20] mm/s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)

    pts = placements(a.episodes, a.placement_seed)
    env_cfg = load_env_cfg(a.task)
    env_cfg.seed = a.sim_seed
    if a.nominal:
        for k in ("joint_stiffness", "joint_damping", "egg_friction", "egg_mass"):
            env_cfg.events.pop(k, None)
        env_cfg.events["episode"].params.update(start_shape_prob=0.0, robot_dz_range=(0.0, 0.0))
    n_env = min(a.envs, len(pts))
    env_cfg.scene.num_envs = n_env
    agent_cfg = load_rl_cfg(a.task)
    base = ManagerBasedRlEnv(env_cfg, device=a.device)
    env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
    if a.agent == "trained":
        runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=a.device)
        runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=a.device)
        policy = runner.get_inference_policy(device=a.device)
    else:
        def policy(obs):
            if a.agent == "hold":
                return mdp.state(base).start_diff / mdp.C.DEFAULT.action.diff_limit_m
            return torch.zeros(n_env, 1, device=a.device)  # zero requests D=0, not hold

    rows = []
    tm = base.termination_manager
    for start in range(0, len(pts), n_env):
        batch = pts[start:start + n_env]
        ov = torch.full((n_env, 2), float("nan"), device=a.device)
        ov[: len(batch)] = torch.tensor(batch, dtype=torch.float32, device=a.device)
        base._kd_uv_override = ov
        obs, _ = base.reset()
        if a.cable_speed_mm_s is not None:
            base.action_manager.get_term(mdp.ACTION_TERM)._vmax.fill_(a.cable_speed_mm_s / 1000)
        obs = env.get_observations()
        start_diff = mdp.state(base).start_diff[:, 0].cpu().numpy().copy()
        done = np.zeros(n_env, dtype=bool)
        done[len(batch):] = True
        decisions = np.zeros(n_env, dtype=int)
        while not done.all():
            with torch.inference_mode():
                act = policy(obs)
            touched_before = mdp.egg_touched(base).cpu().numpy()
            disp_before = (mdp.egg_uv_true(base) - mdp.state(base).ped_uv).norm(dim=-1).cpu().numpy()
            obs, _, dones, _ = env.step(act)
            decisions += ~done
            if int(decisions.max()) % 5 == 0:
                print(f"batch {start//n_env + 1}: decision {int(decisions.max())}, {int((~done).sum())} active", flush=True)
            fired = {k: tm.get_term(k).cpu().numpy() for k in OUTCOMES}
            for i in np.nonzero(dones.cpu().numpy().astype(bool) & ~done)[0]:
                outcome = next((k for k in OUTCOMES if fired[k][i]), "unknown")
                rows.append(dict(episode=start + int(i), u=float(batch[i][0]), v=float(batch[i][1]),
                                 start_diff_mm=round(float(start_diff[i]) * 1e3, 1),
                                 side="+u" if batch[i][0] > 0 else "-u", outcome=outcome,
                                 decisions=int(decisions[i]), touched=bool(touched_before[i] or outcome == "knocked_down"),
                                 egg_offset_mm_before_end=round(float(disp_before[i]) * 1e3, 2)))
                done[i] = True
        print(f"evaluated {min(start + n_env, len(pts))}/{len(pts)}", flush=True)

    rows.sort(key=lambda r: r["episode"])
    with (out / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts = {k: sum(r["outcome"] == k for r in rows) for k in (*OUTCOMES, "unknown")}
    n = len(rows)
    k = counts["knocked_down"]
    valid = n - counts["spontaneous_fall"] - counts["numerical_fault"]
    summary = {
        "task": a.task, "agent": a.agent, "checkpoint": a.checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest() if a.checkpoint else None,
        "episodes": n, "counts": counts,
        "success_rate_all": k / n, "success_wilson95_all": wilson(k, n),
        "valid_episodes": valid, "success_rate_valid": k / valid if valid else None,
        "success_wilson95_valid": wilson(k, valid) if valid else None,
        "acceptance_target": ">= 190/200 successes (95%) on held-out placements",
        "start_shape": "straight" if a.nominal else "random tabulated shape with p=0.7 (the rig continues from where it stopped)",
        "sample_rate_meets_95_percent": k >= math.ceil(0.95 * n),
        "meets_target": n >= 200 and k >= math.ceil(0.95 * n),
        "placement_seed": a.placement_seed, "sim_seed": a.sim_seed, "nominal_dynamics": a.nominal,
        "common_speed_mm_s": a.cable_speed_mm_s,
        "contract_id": mdp.C.CONTRACT_ID,
        "contract_source_sha256": mdp.C.source_file_sha256(),
        "source_model_sha256": sha256(SOURCE_XML),
        "perception_files": {name: sha256(Path(mdp.C.__file__).with_name(name))
                             for name in ("egg_vision.py", "camera_geometry.py")},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "eval_region": {"abs_u_m": EVAL_ABS_U, "v_m": EVAL_V},
        "perception": "rendered_rgb_detector" if "-Camera" in a.task else "geometric_proxy",
        "rendered_camera": camera_manifest() if "-Camera" in a.task else None,
        "median_decisions_success": float(np.median([r["decisions"] for r in rows if r["outcome"] == "knocked_down"]))
        if k else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"counts": counts, "success": f"{k}/{n}", "wilson95": summary["success_wilson95_all"]}))
    env.close()
    try:
        _plot(rows, out / "placements.png")
    except Exception as exc:  # plotting is optional
        print("plot skipped:", exc)


def _plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"knocked_down": "#2a9d8f", "time_out": "#e76f51", "egg_out_of_bounds": "#f4a261",
              "spontaneous_fall": "#8d99ae", "numerical_fault": "#000000", "unknown": "#9b5de5"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for k, c in colors.items():
        pts = [(r["u"], r["v"]) for r in rows if r["outcome"] == k]
        if pts:
            x, y = zip(*pts)
            ax.scatter(np.array(x) * 100, np.array(y) * 100, s=18, c=c, label=f"{k} ({len(pts)})")
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("u [cm] (lateral, frame C)")
    ax.set_ylabel("v [cm] (along rest backbone from joint 2)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
