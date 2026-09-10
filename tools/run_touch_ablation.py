"""Train and evaluate a paired TCH-v1 experiment, defaulting to seed 42.

Use --seeds 42 43 44 45 46 47 for the predeclared replication set.
Use --evaluate-only to finish evaluation of already completed training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def run(cmd, log):
    # Preserve a complete console log and live progress, including errors.
    with log.open("w") as f:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
        if proc.wait():
            raise RuntimeError(f"Command failed; inspect {log}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--logger", choices=("wandb", "tensorboard"), default="wandb")
    p.add_argument("--evaluate-only", action="store_true")
    args = p.parse_args()
    from spirob_mjlab.touch_ablation import TASK_IDS, PROTOCOL, SEEDS, CHECKPOINT_INDICES
    if len(set(args.seeds)) != len(args.seeds) or any(s not in SEEDS for s in args.seeds):
        p.error("Use unique predeclared seeds from 42 through 47")
    if git("status", "--porcelain"):
        p.error("Work from a clean committed branch")
    commit = git("rev-parse", "HEAD")
    root = Path(args.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "tools/check_touch_ablation.py", "--device", "cuda:0"],
        root / "preflight.log")
    for seed in args.seeds:
        # Counterbalance machine-time order; seed 42 begins with touch.
        arms = ("touch", "no-touch") if seed % 2 == 0 else ("no-touch", "touch")
        for arm in arms:
            folder = root / f"seed{seed}" / arm
            manifest_file = folder / "run.json"
            if args.evaluate_only:
                manifest = json.loads(manifest_file.read_text())
                if manifest["git_commit"] != commit or not manifest["training_complete"]:
                    raise RuntimeError("Incomplete training or source mismatch")
                if sha(REPO / "uv.lock") != manifest["dependency_lock_sha256"]:
                    raise RuntimeError("Lockfile mismatch")
            else:
                if folder.exists():
                    raise FileExistsError(f"Preserving existing run: {folder}; use --evaluate-only if trained")
                folder.mkdir(parents=True)
                manifest = {
                    "protocol": PROTOCOL, "arm": arm, "training_seed": seed,
                    "git_commit": commit, "base_commit": "8e985fffdf60cd6c0ba453f053bc1d3b6cd9e654",
                    "dependency_lock_sha256": sha(REPO / "uv.lock"),
                    "num_envs": 256, "iterations": 500, "steps_per_env": 32,
                    "total_transitions": 4096000, "resume": False,
                    "training_complete": False, "evaluation_complete": False,
                    "checkpoints": {},
                }
                cmd = [sys.executable, "-m", "mjlab.scripts.train", TASK_IDS[arm],
                       "--env.scene.num-envs", "256", "--env.seed", str(seed),
                       "--agent.seed", str(seed), "--agent.max-iterations", "500",
                       "--agent.resume", "False", "--agent.logger", args.logger,
                       "--agent.wandb-project", "mjlab", "--agent.run-name", f"tch-v1-{arm}-seed{seed}",
                       "--log-root", str(folder / "training")]
                manifest["command"] = cmd
                manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
                start = time.monotonic()
                run(cmd, folder / "train.log")
                manifest["training_wall_seconds"] = time.monotonic() - start
                for index in CHECKPOINT_INDICES:
                    candidates = list((folder / "training").rglob(f"model_{index}.pt"))
                    if len(candidates) != 1:
                        raise RuntimeError(f"Expected exactly one model_{index}.pt in {folder}")
                    checkpoint = candidates[0]
                    manifest["checkpoints"][str(index)] = {"path": str(checkpoint), "sha256": sha(checkpoint)}
                audits = list((folder / "training").rglob("touch_initial_state.json"))
                if len(audits) != 1:
                    raise RuntimeError("Missing initial model audit")
                manifest["initial_model"] = json.loads(audits[0].read_text())
                manifest["training_complete"] = True
                manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
            for index in CHECKPOINT_INDICES:
                output = folder / "evaluation" / f"model_{index}"
                if (output / "summary.json").exists():
                    # Never silently accept altered or incomplete prior evidence.
                    for line in (output / "manifest.sha256").read_text().splitlines():
                        expected, name = line.split("  ", 1)
                        if sha(output / name) != expected:
                            raise RuntimeError(f"Evidence hash mismatch: {output / name}")
                    continue
                checkpoint = manifest["checkpoints"][str(index)]
                run([sys.executable, "tools/evaluate_touch_ablation.py", "--arm", arm,
                     "--run-id", f"tch-v1-{arm}-seed{seed}", "--training-seed", str(seed),
                     "--training-git-commit", commit, "--checkpoint-file", checkpoint["path"],
                     "--expected-sha256", checkpoint["sha256"], "--checkpoint-index", str(index),
                     "--output-dir", str(output), "--repeats", "20" if index == 499 else "5"],
                    folder / f"evaluate_model_{index}.log")
            manifest["evaluation_complete"] = True
            manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
        audits = [json.loads((root / f"seed{seed}" / a / "run.json").read_text())["initial_model"] for a in arms]
        if audits[0] != audits[1]:
            raise RuntimeError(f"Paired initial networks differ for seed {seed}; do not interpret as a controlled pair")
    report = root / ("report-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run([sys.executable, "tools/analyze_touch_ablation.py", "--input-root", str(root),
         "--output-dir", str(report)], root / "analysis.log")
    print(f"Training, evaluation and report complete: {report}")


if __name__ == "__main__":
    main()
