"""Validate and compare the three frozen-checkpoint S1 episode CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


PROTOCOL_VERSION = "S1-v1"
EXPECTED_TRAINING_SEEDS = {42, 43, 44}
GRID_OFFSETS_MM = (-10, -5, 0, 5, 10)
EXPECTED_GRID = {(dx, dy) for dy in GRID_OFFSETS_MM for dx in GRID_OFFSETS_MM}


def _truth(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


@dataclass(frozen=True)
class CheckpointResult:
    csv_path: Path
    run_id: str
    training_seed: int
    training_git_commit: str
    checkpoint_name: str
    checkpoint_sha256: str
    evaluation_seed: int
    nominal_success_rows: int
    nominal_trajectory_fingerprints: int
    grid_success_points: int
    grid_fall_points: int
    grid_timeout_points: int
    oob_terminations: int
    nan_terminations: int
    overlap_rows: int
    video_present: bool
    nominal_complete: bool
    grid_complete: bool
    gate_pass: bool
    rows: tuple[dict[str, str], ...]


def _single(values: set[str], label: str, csv_path: Path) -> str:
    if len(values) != 1:
        raise ValueError(
            f"{csv_path}: expected one {label}, found {sorted(values)}"
        )
    return next(iter(values))


def _read_checkpoint(csv_path: Path) -> CheckpointResult:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = tuple(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{csv_path}: no episode rows")

    protocol = _single({row["protocol_version"] for row in rows}, "protocol", csv_path)
    if protocol != PROTOCOL_VERSION:
        raise ValueError(
            f"{csv_path}: expected {PROTOCOL_VERSION}, found {protocol}"
        )
    mode = _single({row["policy_mode"] for row in rows}, "policy mode", csv_path)
    if mode != "deterministic":
        raise ValueError(f"{csv_path}: S1 requires deterministic policy output")

    run_id = _single({row["run_id"] for row in rows}, "run ID", csv_path)
    seed = int(_single({row["training_seed"] for row in rows}, "training seed", csv_path))
    training_commit = _single(
        {row["training_git_commit"] for row in rows}, "training commit", csv_path
    )
    checkpoint_name = _single(
        {row["checkpoint_name"] for row in rows}, "checkpoint name", csv_path
    )
    checkpoint_sha = _single(
        {row["checkpoint_sha256"] for row in rows}, "checkpoint SHA-256", csv_path
    )
    evaluation_seed = int(
        _single({row["evaluation_seed"] for row in rows}, "evaluation seed", csv_path)
    )

    nominal = [row for row in rows if row["protocol_phase"] == "nominal_repeat"]
    grid = [row for row in rows if row["protocol_phase"] == "grid"]
    observed_grid = {
        (int(row["egg_offset_x_mm"]), int(row["egg_offset_y_mm"]))
        for row in grid
    }
    nominal_conditions = {row["physical_condition_id"] for row in nominal}
    nominal_complete = len(nominal) == 5 and len(nominal_conditions) == 1
    grid_complete = len(grid) == 25 and observed_grid == EXPECTED_GRID

    nominal_success = sum(_truth(row["success"]) for row in nominal)
    grid_success = sum(_truth(row["success"]) for row in grid)
    grid_fall = sum(_truth(row["fall"]) for row in grid)
    grid_timeout = sum(_truth(row["timeout"]) for row in grid)
    oob_count = sum(_truth(row["oob"]) for row in rows)
    nan_count = sum(_truth(row["nan"]) for row in rows)
    overlap_count = sum(_truth(row["termination_overlap"]) for row in rows)

    video_paths = [
        csv_path.parent / row["video_file"]
        for row in nominal
        if row["video_file"].strip()
    ]
    video_present = bool(video_paths) and all(path.is_file() for path in video_paths)
    checkpoint_hash_file_present = (csv_path.parent / "checkpoint.sha256").is_file()

    gate = (
        nominal_complete
        and nominal_success == 5
        and grid_complete
        and oob_count == 0
        and nan_count == 0
        and overlap_count == 0
        and video_present
        and checkpoint_hash_file_present
    )

    return CheckpointResult(
        csv_path=csv_path,
        run_id=run_id,
        training_seed=seed,
        training_git_commit=training_commit,
        checkpoint_name=checkpoint_name,
        checkpoint_sha256=checkpoint_sha,
        evaluation_seed=evaluation_seed,
        nominal_success_rows=nominal_success,
        nominal_trajectory_fingerprints=len(
            {row["trajectory_fingerprint_sha256"] for row in nominal}
        ),
        grid_success_points=grid_success,
        grid_fall_points=grid_fall,
        grid_timeout_points=grid_timeout,
        oob_terminations=oob_count,
        nan_terminations=nan_count,
        overlap_rows=overlap_count,
        video_present=video_present,
        nominal_complete=nominal_complete,
        grid_complete=grid_complete,
        gate_pass=gate,
        rows=rows,
    )


def _make_comparison_figure(results: list[CheckpointResult], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "success": "#2e7d32",
        "fall": "#d84315",
        "timeout": "#757575",
        "oob": "#6a1b9a",
        "nan": "#111111",
        "overlap": "#c2185b",
        "unknown": "#f9a825",
    }
    labels = {
        "success": "S",
        "fall": "F",
        "timeout": "T",
        "oob": "O",
        "nan": "N",
        "overlap": "X",
        "unknown": "?",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), sharex=True, sharey=True)
    for ax, result in zip(axes, results, strict=True):
        grid = [row for row in result.rows if row["protocol_phase"] == "grid"]
        for row in grid:
            raw_outcome = row["outcome"]
            category = "overlap" if raw_outcome.startswith("overlap:") else raw_outcome
            x = int(row["egg_offset_x_mm"])
            y = int(row["egg_offset_y_mm"])
            ax.scatter(
                x,
                y,
                s=600,
                marker="s",
                color=colors.get(category, colors["unknown"]),
                edgecolor="white",
                linewidth=1.2,
            )
            ax.text(
                x,
                y,
                labels.get(category, "?"),
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
        ax.scatter(
            0,
            0,
            s=735,
            marker="s",
            facecolors="none",
            edgecolors="#1565c0",
            linewidth=2.0,
        )
        ax.set_title(
            f"Seed {result.training_seed}: {result.grid_success_points}/25\n"
            f"{result.checkpoint_name}"
        )
        ax.set_xticks(GRID_OFFSETS_MM)
        ax.set_yticks(GRID_OFFSETS_MM)
        ax.set_xlim(-13, 13)
        ax.set_ylim(-13, 13)
        ax.set_aspect("equal")
        ax.grid(True, color="#d0d0d0", linewidth=0.7)
        ax.set_xlabel("Egg x offset (mm)")
    axes[0].set_ylabel("Egg y offset (mm)")
    fig.suptitle(
        "S1 fixed-policy deterministic outcome maps\n"
        "One exact rollout per point; no interpolation and no probability estimate"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    png = output_dir / "three_seed_success_basins.png"
    svg = output_dir / "three_seed_success_basins.svg"
    fig.savefig(png, dpi=200)
    fig.savefig(svg)
    plt.close(fig)
    return [png, svg]


def _write_summary_csv(results: list[CheckpointResult], output_dir: Path) -> Path:
    path = output_dir / "three_seed_summary.csv"
    columns = (
        "run_id",
        "training_seed",
        "training_git_commit",
        "checkpoint_name",
        "checkpoint_sha256",
        "evaluation_seed",
        "nominal_success_rows",
        "nominal_repeat_rows",
        "nominal_physical_conditions",
        "nominal_trajectory_fingerprints",
        "grid_success_points",
        "grid_total_points",
        "grid_fall_points",
        "grid_timeout_points",
        "oob_terminations",
        "nan_terminations",
        "termination_overlap_rows",
        "nominal_video_present",
        "s1_checkpoint_gate_pass",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "run_id": result.run_id,
                    "training_seed": result.training_seed,
                    "training_git_commit": result.training_git_commit,
                    "checkpoint_name": result.checkpoint_name,
                    "checkpoint_sha256": result.checkpoint_sha256,
                    "evaluation_seed": result.evaluation_seed,
                    "nominal_success_rows": result.nominal_success_rows,
                    "nominal_repeat_rows": 5 if result.nominal_complete else "invalid",
                    "nominal_physical_conditions": 1 if result.nominal_complete else "invalid",
                    "nominal_trajectory_fingerprints": result.nominal_trajectory_fingerprints,
                    "grid_success_points": result.grid_success_points,
                    "grid_total_points": 25 if result.grid_complete else "invalid",
                    "grid_fall_points": result.grid_fall_points,
                    "grid_timeout_points": result.grid_timeout_points,
                    "oob_terminations": result.oob_terminations,
                    "nan_terminations": result.nan_terminations,
                    "termination_overlap_rows": result.overlap_rows,
                    "nominal_video_present": result.video_present,
                    "s1_checkpoint_gate_pass": result.gate_pass,
                }
            )
    return path


def _write_report(
    results: list[CheckpointResult],
    overall_gate: bool,
    success_min: int,
    success_max: int,
    output_dir: Path,
) -> Path:
    path = output_dir / "S1_REPORT.md"
    table_rows = [
        "| Seed | Checkpoint | Nominal | Grid success points | Fall | Timeout | OOB | NaN | Gate |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        table_rows.append(
            f"| {result.training_seed} | `{result.checkpoint_name}` | "
            f"{result.nominal_success_rows}/5 | {result.grid_success_points}/25 | "
            f"{result.grid_fall_points} | {result.grid_timeout_points} | "
            f"{result.oob_terminations} | {result.nan_terminations} | "
            f"{'PASS' if result.gate_pass else 'FAIL'} |"
        )

    if overall_gate:
        decision = (
            "**S1 PASSES.** All three final checkpoints place the egg at the "
            "nominal condition, all three 25-point maps are complete, and no "
            "OOB or NaN termination occurred. The next experiment is controlled "
            "egg-position randomization during training. Keep the robot base fixed."
        )
    else:
        failed = ", ".join(
            str(result.training_seed) for result in results if not result.gate_pass
        )
        decision = (
            f"**S1 FAILS.** Seed(s) {failed} do not satisfy the frozen-task gate. "
            "Stop here and diagnose training repeatability. Do not change the "
            "reward, introduce spawn randomization, or add base x/y motion."
        )

    report = f"""# S1 â€” Frozen-policy evaluation and training repeatability

## Question

Does the unchanged fixed-scene Stage-1 task produce a nominal placement policy
for all three training seeds, and what exact nearby egg positions does each
frozen policy solve?

## Results

{chr(10).join(table_rows)}

The observed deterministic grid coverage ranges from **{success_min}/25 to
{success_max}/25** successful exact positions. This range reports every seed;
it is not replaced by an average.

The five nominal rows per seed are computational repeats of one physical
condition. They are not five independent statistical trials. Each grid cell is
one deterministic rollout at one exact initial condition; the map is not a
success-probability estimate.

## Decision

{decision}
"""
    path.write_text(report, encoding="utf-8")
    return path


def summarize(csv_paths: list[Path], output_dir: Path) -> dict[str, object]:
    if len(csv_paths) != 3:
        raise ValueError("S1 comparison requires exactly three episode CSV files.")
    results = sorted((_read_checkpoint(path) for path in csv_paths), key=lambda r: r.training_seed)
    observed_seeds = {result.training_seed for result in results}
    if observed_seeds != EXPECTED_TRAINING_SEEDS:
        raise ValueError(
            f"Expected training seeds {sorted(EXPECTED_TRAINING_SEEDS)}, "
            f"found {sorted(observed_seeds)}"
        )
    evaluation_seeds = {result.evaluation_seed for result in results}
    if len(evaluation_seeds) != 1:
        raise ValueError(
            f"All checkpoints must use the same evaluation seed; found {sorted(evaluation_seeds)}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    success_values = [result.grid_success_points for result in results]
    success_min = min(success_values)
    success_max = max(success_values)
    overall_gate = all(result.gate_pass for result in results)

    summary_csv = _write_summary_csv(results, output_dir)
    figures = _make_comparison_figure(results, output_dir)
    report = _write_report(
        results, overall_gate, success_min, success_max, output_dir
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "training_seeds": [result.training_seed for result in results],
        "evaluation_seed": next(iter(evaluation_seeds)),
        "grid_success_points_by_seed": {
            str(result.training_seed): result.grid_success_points for result in results
        },
        "grid_success_points_range": [success_min, success_max],
        "grid_success_points_range_width": success_max - success_min,
        "failed_training_seeds": [
            result.training_seed for result in results if not result.gate_pass
        ],
        "s1_exit_gate_pass": overall_gate,
        "next_action": (
            "Begin controlled egg-position training randomization with the base fixed."
            if overall_gate
            else "Diagnose training repeatability before changing the task."
        ),
        "generated_files": [
            summary_csv.name,
            report.name,
            *[figure.name for figure in figures],
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Checkpoint-specific episodes.csv; pass exactly three times.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = summarize(
        [Path(value).expanduser().resolve() for value in args.input],
        Path(args.output_dir).expanduser().resolve(),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
