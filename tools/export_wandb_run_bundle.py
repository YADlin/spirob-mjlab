#!/usr/bin/env python3
"""Export one W&B run into a compact, shareable diagnostic ZIP (v1.1).

Requires only the `wandb` package. Pandas is not required.

Examples
--------
python export_wandb_run_bundle.py \
  --run p20242201-bits-pilani-goa-campus/mjlab/xxxxx
  --output-root "file_location"

# Notice that the run argument takes '/' as relative folder path while output argument takes '\'

The --run argument may also be a W&B URL or a path containing `/runs/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import sys
import traceback
import zipfile
from collections.abc import Mapping
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import wandb


IMPORTANT_RUN_FILES = {
    "config.yaml",
    "requirements.txt",
    "wandb-metadata.json",
    "wandb-summary.json",
    "diff.patch",
    "git.patch",
    "metadata.json",
}

PLOT_KEYS = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Episode_Reward/egg_to_bucket_distance",
    "Episode_Reward/reach_egg",
    "Episode_Reward/inside_bucket",
    "Episode_Reward/action_l2",
    "Episode_Termination/success_egg_inside_bucket",
    "Episode_Termination/egg_oob",
    "Episode_Termination/egg_fell",
    "Episode_Termination/time_out",
    "Loss/value",
    "Loss/surrogate",
    "Loss/entropy",
    "Loss/learning_rate",
    "Policy/mean_std",
    "Perf/total_fps",
]


def normalize_run_path(value: str) -> str:
    """Accept entity/project/id, a URL, or entity/project/runs/id."""
    value = value.strip()
    if "://" in value:
        parsed = urlparse(value)
        value = parsed.path

    parts = [p for p in value.strip("/").split("/") if p]
    if "runs" in parts:
        idx = parts.index("runs")
        if idx < 2 or idx + 1 >= len(parts):
            raise ValueError(f"Could not parse W&B run path: {value}")
        parts = [parts[idx - 2], parts[idx - 1], parts[idx + 1]]

    if len(parts) != 3:
        raise ValueError(
            "Run must resolve to entity/project/run_id. "
            f"Received: {value!r} -> {parts!r}"
        )
    return "/".join(parts)


def _direct_attribute(value: Any, name: str) -> Any:
    """Read a real attribute without triggering SDK ``__getattr__`` fallbacks.

    W&B ``Summary`` objects interpret unknown attribute names as dictionary
    keys.  Therefore ``hasattr(summary, "item")`` raises ``KeyError`` instead
    of returning ``False``.  ``object.__getattribute__`` avoids that trap.
    """
    try:
        return object.__getattribute__(value, name)
    except (AttributeError, KeyError, TypeError):
        return None


def json_safe(value: Any) -> Any:
    """Convert W&B SDK, numpy-like, and standard values into JSON data."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return json_safe(value.value)

    # Unwrap W&B Summary/SummarySubDict and similar SDK proxy objects first.
    # Their ``__getattr__`` treats names such as ``item`` as dictionary keys.
    for attr_name in ("_dict", "_json_dict"):
        inner = _direct_attribute(value, attr_name)
        if inner is not None and inner is not value:
            return json_safe(inner)

    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]

    # Handle numpy scalars/arrays without calling ``hasattr`` on SDK objects.
    value_type = type(value)
    module_name = getattr(value_type, "__module__", "")
    if module_name.startswith("numpy"):
        item_method = getattr(value_type, "item", None)
        if callable(item_method):
            try:
                return json_safe(item_method(value))
            except Exception:
                pass
        tolist_method = getattr(value_type, "tolist", None)
        if callable(tolist_method):
            try:
                return json_safe(tolist_method(value))
            except Exception:
                pass

    return repr(value)


def scalar_for_csv(value: Any) -> Any:
    value = json_safe(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def write_json_optional(path: Path, value_factory) -> bool:
    """Write an optional section and continue if the W&B SDK rejects it."""
    try:
        value = value_factory()
        write_json(path, value)
        return True
    except Exception:
        error_path = path.with_name(path.stem + "_error.txt")
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        return False


def write_history(rows: list[dict[str, Any]], output_dir: Path) -> None:
    jsonl_path = output_dir / "history_full.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")

    keys: set[str] = set()
    for row in rows:
        keys.update(str(k) for k in row.keys())

    preferred = ["_step", "_timestamp", "_runtime"]
    fieldnames = [k for k in preferred if k in keys] + sorted(keys.difference(preferred))

    csv_path = output_dir / "history_full.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({str(k): scalar_for_csv(v) for k, v in row.items()})


def numeric_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def produce_diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_keys = sorted({str(k) for row in rows for k in row.keys()})
    stats: dict[str, Any] = {}
    for key in all_keys:
        vals = numeric_values(rows, key)
        if not vals:
            continue
        stats[key] = {
            "count": len(vals),
            "first": vals[0],
            "last": vals[-1],
            "min": min(vals),
            "max": max(vals),
            "mean": statistics.fmean(vals),
            "median": statistics.median(vals),
        }

    warnings: list[str] = []
    oob = numeric_values(rows, "Episode_Termination/egg_oob")
    episode_len = numeric_values(rows, "Train/mean_episode_length")
    success = numeric_values(rows, "Episode_Termination/success_egg_inside_bucket")
    learning_rate = numeric_values(rows, "Loss/learning_rate")
    policy_std = numeric_values(rows, "Policy/mean_std")

    if oob and statistics.median(oob) > 1:
        warnings.append(
            "Episode_Termination/egg_oob has a median greater than 1. "
            "This often means many parallel environments terminate out-of-bounds each iteration."
        )
    if episode_len and statistics.median(episode_len) <= 2:
        warnings.append(
            "Train/mean_episode_length has a median <= 2. Episodes are ending almost immediately, "
            "so meaningful policy learning is unlikely."
        )
    if success and max(success) == 0:
        warnings.append("No successful egg-inside-bucket termination was logged.")
    if learning_rate and max(learning_rate) / max(min(learning_rate), 1e-12) > 100:
        warnings.append("Learning rate varies by more than 100x; inspect adaptive-KL scheduling/update stability.")
    if policy_std and policy_std[-1] > 1.5 * policy_std[0]:
        warnings.append("Policy mean standard deviation increased substantially during the run.")

    return {
        "row_count": len(rows),
        "history_keys": all_keys,
        "metric_statistics": stats,
        "automatic_warnings": warnings,
    }


def make_plots(rows: list[dict[str, Any]], plot_dir: Path) -> tuple[bool, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return False, f"matplotlib unavailable: {exc}"

    plot_dir.mkdir(parents=True, exist_ok=True)
    x_default = list(range(len(rows)))
    created = 0

    for key in PLOT_KEYS:
        xs: list[float] = []
        ys: list[float] = []
        for idx, row in enumerate(rows):
            value = row.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                step = row.get("_step", x_default[idx])
                try:
                    x_value = float(step)
                except Exception:
                    x_value = float(idx)
                xs.append(x_value)
                ys.append(float(value))
        if not ys:
            continue

        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        ax.plot(xs, ys)
        ax.set_title(key)
        ax.set_xlabel("W&B step")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
        fig.savefig(plot_dir / f"{safe_name}.png", dpi=160)
        plt.close(fig)
        created += 1

    return True, f"created {created} plots"


def download_important_files(run: Any, output_dir: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    files_dir = output_dir / "run_files"
    files_dir.mkdir(parents=True, exist_ok=True)

    for file_obj in run.files(per_page=1000):
        name = str(getattr(file_obj, "name", ""))
        entry = {
            "name": name,
            "size": getattr(file_obj, "size", None),
            "md5": getattr(file_obj, "md5", None),
            "url": getattr(file_obj, "url", None),
            "downloaded": False,
            "error": None,
        }
        try:
            basename = Path(name).name
            if basename in IMPORTANT_RUN_FILES or name.endswith((".yaml", ".yml", ".json", ".patch")):
                file_obj.download(root=str(files_dir), replace=True)
                entry["downloaded"] = True
        except Exception as exc:
            entry["error"] = repr(exc)
        inventory.append(json_safe(entry))

    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        help="W&B run as entity/project/run_id, a W&B URL, or a path containing /runs/.",
    )
    parser.add_argument(
        "--output-root",
        default="wandb_exports",
        help="Directory in which the export folder and ZIP will be created.",
    )
    args = parser.parse_args()

    run_path = normalize_run_path(args.run)
    api = wandb.Api()
    run = api.run(run_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / f"wandb_{run.id}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    print(f"Fetching complete scalar history for {run_path} ...")
    rows = [dict(row) for row in run.scan_history(page_size=1000)]
    rows.sort(key=lambda r: (r.get("_step", -1), r.get("_timestamp", 0)))
    write_history(rows, output_dir)

    run_info = {
        "path": run_path,
        "id": run.id,
        "name": run.name,
        "url": run.url,
        "state": run.state,
        "entity": run.entity,
        "project": run.project,
        "created_at": getattr(run, "created_at", None),
        "user": getattr(run, "user", None),
        "notes": getattr(run, "notes", None),
        "tags": getattr(run, "tags", None),
        "last_history_step": getattr(run, "lastHistoryStep", None),
    }
    write_json(output_dir / "run_info.json", run_info)
    write_json_optional(output_dir / "config.json", lambda: dict(run.config))
    write_json_optional(output_dir / "summary.json", lambda: dict(run.summary))
    write_json_optional(output_dir / "metadata.json", lambda: getattr(run, "metadata", None))
    write_json_optional(
        output_dir / "system_metrics_latest.json",
        lambda: getattr(run, "system_metrics", None),
    )

    # System stream is sampled, but still useful for GPU/CPU diagnostics.
    try:
        system_rows = run.history(samples=10000, pandas=False, stream="system")
        if system_rows:
            system_dir = output_dir / "system_history"
            system_dir.mkdir(parents=True, exist_ok=True)
            write_history([dict(row) for row in system_rows], system_dir)
    except Exception as exc:
        (output_dir / "system_history_error.txt").write_text(traceback.format_exc(), encoding="utf-8")

    inventory = download_important_files(run, output_dir)
    write_json(output_dir / "run_files_inventory.json", inventory)

    # Ask W&B for parquet history exports when the server supports them.
    parquet_status: dict[str, Any] = {"attempted": False, "success": False, "error": None}
    history_export_method = _direct_attribute(run, "download_history_exports")
    if callable(history_export_method):
        parquet_status["attempted"] = True
        try:
            parquet_dir = output_dir / "history_parquet"
            parquet_dir.mkdir(parents=True, exist_ok=True)
            result = history_export_method(parquet_dir, require_complete_history=False)
            parquet_status["success"] = True
            parquet_status["result"] = repr(result)
        except Exception as exc:
            parquet_status["error"] = repr(exc)
    write_json(output_dir / "parquet_export_status.json", parquet_status)

    diagnostic_summary = produce_diagnostic_summary(rows)
    write_json(output_dir / "diagnostic_summary.json", diagnostic_summary)

    plot_ok, plot_message = make_plots(rows, output_dir / "plots")
    (output_dir / "plot_status.txt").write_text(
        f"success={plot_ok}\n{plot_message}\n", encoding="utf-8"
    )

    readme = f"""W&B diagnostic export
=====================

Run: {run_path}
Name: {run.name}
URL: {run.url}
State: {run.state}
Exported UTC: {timestamp}
History rows: {len(rows)}

Primary files for review:
- history_full.csv: complete scalar history from Run.scan_history()
- history_full.jsonl: lossless row-oriented history
- config.json: hyperparameters/configuration captured by W&B
- summary.json: final/best summary metrics
- run_info.json: run identity, URL, state, notes and tags
- metadata.json: W&B machine/run metadata when available
- diagnostic_summary.json: automatic statistics and warnings
- plots/: quick unsmoothed PNG plots when matplotlib is installed
- run_files/: important YAML/JSON/patch files uploaded with the run
- run_files_inventory.json: inventory of all run files
- history_parquet/: server-provided history exports when available

Upload this ZIP to ChatGPT. Do not include or share your W&B API key.
"""
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")

    zip_path = output_root / f"{output_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in output_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(output_dir.parent))

    print(f"Export folder: {output_dir}")
    print(f"Share this ZIP: {zip_path}")

    warnings = diagnostic_summary.get("automatic_warnings", [])
    if warnings:
        print("\nAutomatic warnings:")
        for warning in warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
