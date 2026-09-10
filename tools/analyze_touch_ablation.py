"""Validate TCH-v1 evidence and produce seed-level tables, figures and a report.

Requires numpy, scipy and matplotlib (already in the pinned MJLab environment).
One seed pair is always exploratory. Repeated fixed-scene episodes are never
treated as independent training replicates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ("touch", "no-touch")
INDICES = (0, 100, 200, 300, 400, 499)
MARGIN = 0.05


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_interval(differences):
    d = np.asarray(differences, dtype=float)
    if d.size < 2 or not np.isfinite(d).all():
        return None
    sd = d.std(ddof=1)
    if sd < 1e-12:
        # Zero observed variability with very few discrete runs does not
        # establish an infinitely precise population effect.
        return None
    half = float(t.ppf(0.975, len(d) - 1) * sd / np.sqrt(len(d)))
    return [max(-1.0, float(d.mean()) - half), min(1.0, float(d.mean()) + half)]


def sign_flip_p(differences):
    d = np.asarray(differences, dtype=float)
    obs = abs(d.mean())
    values = [abs(np.mean(d * signs)) for signs in itertools.product((-1, 1), repeat=len(d))]
    return float(np.mean(np.asarray(values) >= obs - 1e-12))


def decision(d, complete, engineering_clean):
    ci = paired_interval(d)
    if not complete:
        return "Exploratory only: the predeclared six seed pairs are not complete."
    if not engineering_clean:
        return "Unresolved: numerical/termination integrity requires review before causal attribution."
    if ci is None:
        return "Unresolved: observed seed differences have no estimable variation; check floor/ceiling effects and report exact outcomes."
    if ci[0] > MARGIN and sign_flip_p(d) < 0.05:
        return "Evidence of a practically useful touch benefit at the fixed spawn within the 500-update budget."
    if ci[1] < -MARGIN and sign_flip_p(d) < 0.05:
        return "Evidence that touch reduces final fixed-spawn performance within this training budget."
    if ci[0] > -MARGIN and ci[1] < MARGIN:
        return "Final fixed-spawn performance is compatible with practical equivalence within ±5 percentage points; inspect learning efficiency separately."
    return "Inconclusive at the predeclared ±5 percentage-point practical margin."


def load_evidence(root):
    manifests = sorted(Path(root).glob("seed*/*/run.json"))
    runs = {}
    provenance = set()
    for path in manifests:
        m = json.loads(path.read_text())
        key = (m["training_seed"], m["arm"])
        if key in runs or m["arm"] not in ARMS or m["protocol"] != "TCH-fixed-v1":
            raise ValueError("Duplicate or incompatible run")
        if not m["training_complete"] or not m["evaluation_complete"] or m["resume"]:
            raise ValueError(f"Incomplete or resumed run: {path}")
        if (m["iterations"], m["num_envs"], m["steps_per_env"]) != (500, 256, 32):
            raise ValueError("Compute budget mismatch")
        provenance.add((m["git_commit"], m["dependency_lock_sha256"]))
        rows = []
        for index in INDICES:
            folder = path.parent / "evaluation" / f"model_{index}"
            hashes = {}
            for line in (folder / "manifest.sha256").read_text().splitlines():
                h, name = line.split("  ", 1)
                hashes[name] = h
                if sha(folder / name) != h:
                    raise ValueError(f"Evidence checksum mismatch: {folder / name}")
            if "episodes.csv" not in hashes or "summary.json" not in hashes:
                raise ValueError("Incomplete evidence manifest")
            with (folder / "episodes.csv").open() as f:
                batch = list(csv.DictReader(f))
            expected = 20 if index == 499 else 5
            if len(batch) != expected or {int(r["episode_index"]) for r in batch} != set(range(expected)):
                raise ValueError("Missing or duplicate episode rows")
            for r in batch:
                if (int(r["training_seed"]), r["arm"]) != key:
                    raise ValueError("Run/episode identity mismatch")
                if r["checkpoint_sha256"] != m["checkpoints"][str(index)]["sha256"]:
                    raise ValueError("Checkpoint mismatch")
                if r["training_git_commit"] != m["git_commit"] or r["dependency_lock_sha256"] != m["dependency_lock_sha256"]:
                    raise ValueError("Provenance mismatch")
                if r["policy_mode"] != "deterministic" or int(r["checkpoint_index"]) != index:
                    raise ValueError("Policy/checkpoint mismatch")
                if int(r["evaluation_seed"]) != 3210 + int(r["episode_index"]):
                    raise ValueError("Evaluation seed mismatch")
                if abs(float(r["egg_offset_x_mm"])) + abs(float(r["egg_offset_y_mm"])) > 1e-12:
                    raise ValueError("Spawn perturbation in fixed-scene evidence")
                xyz = [float(r[f"initial_egg_{a}_m"]) for a in "xyz"]
                if not np.allclose(xyz, (0.05, 0.15, 0.098), rtol=0, atol=1e-6):
                    raise ValueError("Actual egg spawn differs from nominal")
                rows.append(r)
        runs[key] = (m, rows)
    if not runs or len(provenance) != 1:
        raise ValueError("No complete evidence or mixed model/source versions")
    seeds = sorted({s for s, _ in runs})
    for s in seeds:
        if any((s, arm) not in runs for arm in ARMS):
            raise ValueError(f"Missing paired condition for seed {s}")
        if runs[s, ARMS[0]][0]["initial_model"] != runs[s, ARMS[1]][0]["initial_model"]:
            raise ValueError(f"Initial network mismatch for seed {s}")
    return seeds, runs


def analyze(root, output):
    seeds, runs = load_evidence(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scores = {}
    records = []
    engineering_clean = True
    for (seed, arm), (m, rows) in runs.items():
        by_index = {i: [r for r in rows if int(r["checkpoint_index"]) == i] for i in INDICES}
        curve = [np.mean([r["outcome"] == "success" for r in by_index[i]]) for i in INDICES]
        final = by_index[499]
        scores[seed, arm] = np.array(curve)
        engineering_clean &= all(r["outcome"] in ("success", "fall", "timeout") and int(r["raw_sensor_nonfinite_count"]) == 0 for r in rows)
        x = (np.array(INDICES) + 1) * 8192
        auc = float(np.trapezoid(curve, x=x) / (x[-1] - x[0]))
        success_times = [float(r["episode_length_s"]) for r in final if r["outcome"] == "success"]
        records.append({"seed": seed, "arm": arm, "final_success_fraction": curve[-1],
                        "learning_curve_auc": auc,
                        "success_only_median_time_s": float(np.median(success_times)) if success_times else None,
                        "training_wall_seconds": m["training_wall_seconds"],
                        **{f"final_{o}": sum(r["outcome"] == o for r in final) for o in ("success", "fall", "timeout", "oob", "nan")},
                        "final_other": sum(r["outcome"] not in ("success", "fall", "timeout", "oob", "nan") for r in final)})
    delta = [float(scores[s, "touch"][-1] - scores[s, "no-touch"][-1]) for s in seeds]
    result = {"training_seeds": seeds, "n_independent_seed_pairs": len(seeds),
              "mean_paired_final_effect_pp": 100 * float(np.mean(delta)),
              "paired_95pct_t_interval_fraction": paired_interval(delta),
              "two_sided_sign_flip_p": sign_flip_p(delta),
              "decision": decision(delta, seeds == list(range(42, 48)), engineering_clean),
              "engineering_clean": engineering_clean,
              "limitations": "Single fixed spawn, legacy physics, binary touch with other state observations retained. Approximate seed-level t interval; six seeds can still be insufficient. Sign-flip test assumes exchangeability/symmetry under the null. Execution repeats are not independent training samples. No generalization or hardware inference."}
    with (output / "seed_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    colors = {"touch": "#007C91", "no-touch": "#D46A32"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    x = (np.array(INDICES) + 1) * 8192 / 1e6
    for arm in ARMS:
        values = np.stack([scores[s, arm] for s in seeds])
        for v in values:
            ax[0].plot(x, v * 100, color=colors[arm], alpha=0.22, linewidth=1)
        ax[0].plot(x, values.mean(0) * 100, "o-", color=colors[arm], label=arm)
    ax[0].set(xlabel="Training transitions (millions)", ylabel="Fixed-spawn success (%)", ylim=(-3, 103), title="Frozen-policy learning curves")
    ax[0].legend()
    for j, s in enumerate(seeds):
        v = [scores[s, a][-1] * 100 for a in ARMS]
        ax[1].plot([0, 1], v, "o-", alpha=0.65, label=f"seed {s}")
    ax[1].set(xticks=[0, 1], xticklabels=["Touch", "No touch"], ylabel="Final success (%)", ylim=(-3, 103), title=f"Paired training seeds (n={len(seeds)})")
    ax[1].legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"learning_and_pairs.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    outcomes = ("success", "fall", "timeout", "oob", "nan", "other")
    palette = ("#007C91", "#D46A32", "#C1A243", "#8A5792", "#CC3344", "#555555")
    bottom = np.zeros(2)
    for outcome, color in zip(outcomes, palette):
        vals = [100 * np.mean([r[f"final_{outcome}"] / 20 for r in records if r["arm"] == a]) for a in ARMS]
        axes[0].bar(ARMS, vals, bottom=bottom, label=outcome, color=color)
        bottom += vals
    axes[0].set(ylabel="Final execution outcomes (%)", title="Failures remain in the denominator")
    axes[0].legend(fontsize=8)
    activity = []
    for arm in ARMS:
        vals = []
        for s in seeds:
            rows = [r for r in runs[s, arm][1] if int(r["checkpoint_index"]) == 499]
            counts = np.sum([json.loads(r["raw_sensor_active_counts"]) for r in rows], axis=0)
            vals.append(counts / sum(int(r["observed_steps"]) for r in rows))
        activity.append(np.mean(vals, axis=0))
    im = axes[1].imshow(np.array(activity) * 100, aspect="auto", vmin=0, vmax=100, cmap="viridis")
    axes[1].set(yticks=[0, 1], yticklabels=ARMS, xlabel="Raw touch sensor index (0–41)", title="Raw sensor activation, both arms")
    fig.colorbar(im, ax=axes[1], label="Active control samples (%)")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"outcomes_and_touch.{ext}", dpi=180)
    plt.close(fig)
    lines = ["# TCH-v1 — Fixed-spawn touch ablation", "", result["decision"], "",
             f"Mean paired final difference (touch − no touch): **{result['mean_paired_final_effect_pp']:.2f} percentage points**.",
             f"Independent unit: training seed; {len(seeds)} paired seeds. Each final checkpoint has 20 execution repeats at one nominal spawn.", "",
             "| Seed | Touch success | No-touch success | Difference (pp) |", "|---|---:|---:|---:|"]
    for s, d in zip(seeds, delta):
        lines.append(f"| {s} | {100*scores[s,'touch'][-1]:.1f}% | {100*scores[s,'no-touch'][-1]:.1f}% | {100*d:+.1f} |")
    lines += ["", "![Frozen checkpoint curves and seed pairs](learning_and_pairs.png)",
              "Each faint curve is one training seed; bold curves are means. Intermediate checkpoints use five execution repeats; final checkpoints use twenty. No spatial generalization is tested.", "",
              "![Outcomes and raw sensor activity](outcomes_and_touch.png)",
              "Sensor activity is a diagnostic of available contact information, not proof that the actor uses it. The no-touch actor and critic receive zeros even when raw sensors activate.", "",
              f"Approximate paired 95% t interval (fraction units): {result['paired_95pct_t_interval_fraction']}. Two-sided sign-flip p: {result['two_sided_sign_flip_p']:.4f}.",
              "The seed table also reports learning-curve area over the sampled training window, success-only completion time and measured training wall time. AUC is secondary; final checkpoint selection was fixed before training. Success-only time can favor policies that solve only easy executions and must be read with success frequency.", "",
              result["limitations"], "",
              "Results establish neither general touch necessity nor hardware transfer. Both arms may reach a ceiling, both may fail at a floor, or uncertainty may remain wide. Non-significance alone is not equivalence."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(result["decision"])
    print(output / "report.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    analyze(args.input_root, args.output_dir)
