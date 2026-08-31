"""Validate and compare the paired S2-B/S2-C promotion-screen outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


PROTOCOL_VERSION = "S2-C-promotion-v1"
EVALUATION_SEED = 2510
EXPECTED_S2B_SHA256 = (
    "75606f6877518c6ae33a8b1bd605fa411305c42eb5f120c71df6e008cd6b65b4"
)
IMPROVEMENT_MIN = 0.05


@dataclass(frozen=True)
class PromotionSummaryConfig:
    candidate_dir: str
    """Output directory from the S2-C candidate evaluation."""

    baseline_dir: str
    """Output directory from the S2-B baseline evaluation."""

    output_dir: str
    """New directory for paired comparison evidence."""

    candidate_expected_checkpoint_sha256: str
    """Frozen SHA-256 recorded for the S2-C model_499.pt."""

    baseline_expected_checkpoint_sha256: str = EXPECTED_S2B_SHA256
    """Frozen SHA-256 recorded for the accepted S2-B model_499.pt."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(directory: Path) -> None:
    manifest = directory / "manifest.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        path = directory / relative
        if not path.is_file():
            raise FileNotFoundError(f"Manifest file is missing: {path}")
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"Manifest mismatch for {path}: expected {expected}, "
                f"observed {observed}"
            )


def _load(
    directory: Path,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, str]]]:
    _verify_manifest(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    with (directory / "episodes.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return summary, metadata, rows


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


def _expanded(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = [
        row for row in rows
        if row["protocol_phase"] == "promotion_expanded_10mm"
    ]
    return sorted(selected, key=lambda row: int(row["phase_episode_index"]))


def _phase_rate(rows: list[dict[str, str]], phase: str) -> float:
    selected = [row for row in rows if row["protocol_phase"] == phase]
    if not selected:
        return float("nan")
    return sum(_truth(row["success"]) for row in selected) / len(selected)


def _stratum_rates(rows: list[dict[str, str]]) -> np.ndarray:
    values = np.full(25, np.nan)
    for stratum in range(25):
        selected = [row for row in rows if int(row["spatial_stratum"]) == stratum]
        if selected:
            values[stratum] = sum(
                _truth(row["success"]) for row in selected
            ) / len(selected)
    return values.reshape(5, 5)


def _make_figure(
    *,
    baseline_expanded: list[dict[str, str]],
    candidate_expanded: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    output_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline_rate = _phase_rate(baseline_expanded, "promotion_expanded_10mm")
    candidate_rate = _phase_rate(candidate_expanded, "promotion_expanded_10mm")
    core_rate = _phase_rate(candidate_rows, "promotion_core_5mm")
    candidate_matrix = _stratum_rates(candidate_expanded)
    baseline_matrix = _stratum_rates(baseline_expanded)
    difference = 100.0 * (candidate_matrix - baseline_matrix)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.1))
    labels = ["S2-B\n±10 mm", "S2-C\n±10 mm", "S2-C\n±5 mm"]
    rates = [baseline_rate, candidate_rate, core_rate]
    colors = ["#8a94a6", "#167d9a", "#2e7d32"]
    axes[0].bar(labels, rates, color=colors, width=0.62)
    axes[0].axhline(0.85, color="#b45309", linestyle="--", linewidth=1.4)
    axes[0].axhline(0.90, color="#2e7d32", linestyle=":", linewidth=1.4)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Observed success fraction")
    axes[0].set_title("Frozen-policy promotion screen")
    for index, rate in enumerate(rates):
        axes[0].text(index, rate + 0.025, f"{100.0 * rate:.1f}%", ha="center")

    image = axes[1].imshow(
        difference,
        origin="lower",
        cmap="RdBu",
        vmin=-50.0,
        vmax=50.0,
        interpolation="nearest",
    )
    for y_index in range(5):
        for x_index in range(5):
            axes[1].text(
                x_index,
                y_index,
                f"{difference[y_index, x_index]:+.0f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    axes[1].set_title("S2-C minus S2-B at paired ±10 mm positions")
    axes[1].set_xlabel("Spatial-stratum x index")
    axes[1].set_ylabel("Spatial-stratum y index")
    axes[1].set_xticks(range(5))
    axes[1].set_yticks(range(5))
    fig.colorbar(image, ax=axes[1], label="Success difference (percentage points)")
    fig.text(
        0.5,
        0.01,
        "Promotion-screen estimates from distinct balanced positions; not a "
        "deployment-grade workspace map.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    png = output_dir / "s2c_promotion_comparison.png"
    svg = output_dir / "s2c_promotion_comparison.svg"
    fig.savefig(png, dpi=200)
    fig.savefig(svg)
    plt.close(fig)
    return [png, svg]


def _write_manifest(output_dir: Path) -> Path:
    manifest = output_dir / "manifest.sha256"
    files = sorted(
        path for path in output_dir.rglob("*")
        if path.is_file() and path != manifest
    )
    manifest.write_text(
        "\n".join(
            f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}"
            for path in files
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def summarize(cfg: PromotionSummaryConfig) -> Path:
    for label, value in (
        ("candidate", cfg.candidate_expected_checkpoint_sha256),
        ("baseline", cfg.baseline_expected_checkpoint_sha256),
    ):
        invalid_character = any(
            character not in "0123456789abcdefABCDEF" for character in value
        )
        if len(value) != 64 or invalid_character:
            raise ValueError(f"{label} expected checkpoint SHA-256 is invalid")

    candidate_dir = Path(cfg.candidate_dir).expanduser().resolve()
    baseline_dir = Path(cfg.baseline_dir).expanduser().resolve()
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_summary, candidate_metadata, candidate_rows = _load(candidate_dir)
    baseline_summary, baseline_metadata, baseline_rows = _load(baseline_dir)

    for label, summary, metadata, expected_role, expected_hash in (
        (
            "candidate",
            candidate_summary,
            candidate_metadata,
            "candidate",
            cfg.candidate_expected_checkpoint_sha256,
        ),
        (
            "baseline",
            baseline_summary,
            baseline_metadata,
            "baseline",
            cfg.baseline_expected_checkpoint_sha256,
        ),
    ):
        if summary.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"{label}: wrong protocol version")
        if summary.get("role") != expected_role:
            raise ValueError(f"{label}: wrong role")
        if summary.get("checkpoint_sha256") != expected_hash:
            raise ValueError(f"{label}: unexpected checkpoint hash")
        configuration = metadata.get("configuration", {})
        if configuration.get("evaluation_seed") != EVALUATION_SEED:
            raise ValueError(f"{label}: wrong evaluation seed")
        if not summary.get("protocol_complete"):
            raise ValueError(f"{label}: protocol is incomplete")

    candidate_mode = candidate_summary.get("mode")
    baseline_mode = baseline_summary.get("mode")
    if candidate_mode != baseline_mode:
        raise ValueError("Candidate and baseline modes differ")

    candidate_expanded = _expanded(candidate_rows)
    baseline_expanded = _expanded(baseline_rows)
    if len(candidate_expanded) != len(baseline_expanded):
        raise ValueError("Paired expanded-position row counts differ")

    paired_rows: list[dict[str, object]] = []
    position_mismatches = 0
    candidate_successes = 0
    baseline_successes = 0
    paired_differences: list[float] = []
    candidate_only_successes = 0
    baseline_only_successes = 0
    for candidate, baseline in zip(
        candidate_expanded, baseline_expanded, strict=True
    ):
        candidate_position = (
            int(candidate["phase_episode_index"]),
            int(candidate["spatial_stratum"]),
            float(candidate["egg_offset_x_mm"]),
            float(candidate["egg_offset_y_mm"]),
        )
        baseline_position = (
            int(baseline["phase_episode_index"]),
            int(baseline["spatial_stratum"]),
            float(baseline["egg_offset_x_mm"]),
            float(baseline["egg_offset_y_mm"]),
        )
        if candidate_position != baseline_position:
            position_mismatches += 1
        candidate_success = _truth(candidate["success"])
        baseline_success = _truth(baseline["success"])
        candidate_successes += candidate_success
        baseline_successes += baseline_success
        difference = float(candidate_success) - float(baseline_success)
        paired_differences.append(difference)
        candidate_only_successes += candidate_success and not baseline_success
        baseline_only_successes += baseline_success and not candidate_success
        paired_rows.append(
            {
                "phase_episode_index": candidate_position[0],
                "spatial_stratum": candidate_position[1],
                "offset_x_mm": candidate_position[2],
                "offset_y_mm": candidate_position[3],
                "s2b_baseline_success": baseline_success,
                "s2c_candidate_success": candidate_success,
                "candidate_minus_baseline": difference,
            }
        )

    if position_mismatches:
        raise ValueError(
            "Candidate and baseline position lists differ in "
            f"{position_mismatches} rows"
        )

    total = len(paired_rows)
    candidate_rate = candidate_successes / total
    baseline_rate = baseline_successes / total
    improvement = candidate_rate - baseline_rate
    differences = np.asarray(paired_differences, dtype=np.float64)
    if total > 1:
        half_width = 1.959963984540054 * float(
            np.std(differences, ddof=1) / math.sqrt(total)
        )
        improvement_interval = (
            max(-1.0, improvement - half_width),
            min(1.0, improvement + half_width),
        )
    else:
        improvement_interval = (float("nan"), float("nan"))

    candidate_lower, candidate_upper = _wilson(candidate_successes, total)
    baseline_lower, baseline_upper = _wilson(baseline_successes, total)
    engineering_keys = (
        "oob_terminations_all_rows",
        "nan_terminations_all_rows",
        "termination_overlap_rows",
        "unknown_outcome_rows",
    )
    candidate_engineering_clean = all(
        candidate_summary.get(key) == 0 for key in engineering_keys
    )
    baseline_engineering_clean = all(
        baseline_summary.get(key) == 0 for key in engineering_keys
    )
    smoke_validation_pass = (
        position_mismatches == 0
        and candidate_engineering_clean
        and baseline_engineering_clean
    )
    gate_applicable = candidate_mode == "full"
    candidate_component_pass = candidate_summary.get(
        "candidate_component_gate_pass"
    )
    promotion_pass: bool | None = None
    if gate_applicable:
        promotion_pass = (
            bool(candidate_component_pass)
            and baseline_engineering_clean
            and improvement >= IMPROVEMENT_MIN
        )

    paired_csv = output_dir / "paired_expanded_10mm_conditions.csv"
    with paired_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    figure_paths = _make_figure(
        baseline_expanded=baseline_expanded,
        candidate_expanded=candidate_expanded,
        candidate_rows=candidate_rows,
        output_dir=output_dir,
    )
    comparison = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": candidate_mode,
        "candidate_checkpoint_sha256": candidate_summary["checkpoint_sha256"],
        "baseline_checkpoint_sha256": baseline_summary["checkpoint_sha256"],
        "paired_position_rows": total,
        "paired_position_mismatches": position_mismatches,
        "candidate_expanded_10mm_successes": candidate_successes,
        "candidate_expanded_10mm_success_rate": candidate_rate,
        "candidate_expanded_10mm_wilson_95_lower": candidate_lower,
        "candidate_expanded_10mm_wilson_95_upper": candidate_upper,
        "baseline_expanded_10mm_successes": baseline_successes,
        "baseline_expanded_10mm_success_rate": baseline_rate,
        "baseline_expanded_10mm_wilson_95_lower": baseline_lower,
        "baseline_expanded_10mm_wilson_95_upper": baseline_upper,
        "paired_improvement_fraction": improvement,
        "paired_improvement_percentage_points": 100.0 * improvement,
        "paired_improvement_approx_95_lower": improvement_interval[0],
        "paired_improvement_approx_95_upper": improvement_interval[1],
        "candidate_only_successes": candidate_only_successes,
        "baseline_only_successes": baseline_only_successes,
        "candidate_engineering_clean": candidate_engineering_clean,
        "baseline_engineering_clean": baseline_engineering_clean,
        "smoke_validation_pass": smoke_validation_pass,
        "candidate_component_gate_pass": candidate_component_pass,
        "promotion_gate_applicable": gate_applicable,
        "checkpoint_s2c_promotion_gate_pass": promotion_pass,
        "gate_definition": {
            "candidate_nominal_success_minimum": "42/50",
            "candidate_core_5mm_success_minimum": "90%",
            "candidate_expanded_10mm_success_minimum": "85%",
            "candidate_improvement_over_baseline_minimum": "5 percentage points",
            "candidate_oob": 0,
            "candidate_nan": 0,
            "candidate_overlap": 0,
            "candidate_unknown": 0,
            "baseline_oob": 0,
            "baseline_nan": 0,
            "baseline_overlap": 0,
            "baseline_unknown": 0,
        },
        "interpretation_note": (
            "Candidate and baseline used the same distinct +/-10 mm positions. "
            "This compact screen supports a curriculum-promotion decision; it is "
            "not a deployment-grade workspace certification."
        ),
    }
    summary_path = output_dir / "promotion_summary.json"
    summary_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    _write_manifest(output_dir)

    print(f"[S2-C promotion] Paired conditions: {total}")
    print(f"[S2-C promotion] Position mismatches: {position_mismatches}")
    print(f"[S2-C promotion] S2-B +/-10 mm: {100.0 * baseline_rate:.2f}%")
    print(f"[S2-C promotion] S2-C +/-10 mm: {100.0 * candidate_rate:.2f}%")
    print(f"[S2-C promotion] Improvement: {100.0 * improvement:+.2f} pp")
    print(f"[S2-C promotion] Smoke validation: {smoke_validation_pass}")
    print(f"[S2-C promotion] Final gate: {promotion_pass}")
    print(f"[S2-C promotion] Summary: {summary_path}")
    print(f"[S2-C promotion] Figure: {figure_paths[0]}")
    return output_dir


def main() -> None:
    summarize(tyro.cli(PromotionSummaryConfig))


if __name__ == "__main__":
    main()
