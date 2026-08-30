#!/usr/bin/env python3
"""Compile S2 and verify its paired, balanced, collision-screened resets.

This is a pre-training gate. It does not evaluate manipulation success.
"""

from __future__ import annotations

import argparse

import torch

import spirob_mjlab  # noqa: F401  # Register package tasks.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from spirob_mjlab import mdp

TASK_PROFILES = {
    "Mjlab-SpiRob-EggToBucket-Stage2": {
        "kind": "single_range",
        "range_m": 0.002,
        "nominal_every_n": 4,
    },
    "Mjlab-SpiRob-EggToBucket-Stage2B": {
        "kind": "single_range",
        "range_m": 0.005,
        "nominal_every_n": 5,
    },
    "Mjlab-SpiRob-EggToBucket-Stage2C": {
        "kind": "mixed_range",
        "range_m": 0.010,
        "core_range_m": 0.005,
        "group_fractions": (0.10, 0.30, 0.60),
    },
}
ROBOT_CLEARANCE_M = 0.038
BUCKET_CLEARANCE_M = 0.055
BUCKET_XY = torch.tensor((-0.05, 0.15))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--task-id",
        choices=tuple(TASK_PROFILES),
        default="Mjlab-SpiRob-EggToBucket-Stage2",
    )
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--resets", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    args = parse_args()
    profile = TASK_PROFILES[args.task_id]
    range_m = float(profile["range_m"])
    nominal_every_n = int(profile.get("nominal_every_n", 10))
    require(args.num_envs >= 25, "Use at least 25 environments to cover all strata")
    require(
        args.resets >= nominal_every_n,
        f"Use at least {nominal_every_n} resets to check nominal retention",
    )

    cfg = load_env_cfg(args.task_id)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)

    all_offsets = []
    all_nominal = []
    all_strata = []
    all_rejections = []
    all_core = []
    all_expanded = []
    max_pair_error = 0.0
    min_robot_clearance = float("inf")
    min_bucket_clearance = float("inf")

    try:
        for _ in range(args.resets):
            env.reset()
            egg = env.scene["egg"]
            pedestal = env.scene["pedestal"]
            origins = env.scene.env_origins[:, :2]

            egg_offset = (
                egg.data.root_link_pos_w[:, :2]
                - origins
                - egg.data.default_root_state[:, :2]
            )
            pedestal_offset = (
                pedestal.data.root_link_pos_w[:, :2]
                - origins
                - pedestal.data.default_root_state[:, :2]
            )
            pair_error = torch.max(torch.abs(egg_offset - pedestal_offset)).item()
            max_pair_error = max(max_pair_error, pair_error)

            nominal = mdp.stage2_spawn_is_nominal(env).bool()
            stratum = mdp.stage2_spawn_stratum(env).to(torch.long)
            rejections = mdp.stage2_spawn_rejection_count(env).to(torch.long)
            all_offsets.append(egg_offset.detach().cpu())
            all_nominal.append(nominal.detach().cpu())
            all_strata.append(stratum.detach().cpu())
            all_rejections.append(rejections.detach().cpu())
            if profile["kind"] == "mixed_range":
                all_core.append(
                    mdp.stage2_spawn_is_core_5mm(env).bool().detach().cpu()
                )
                all_expanded.append(
                    mdp.stage2_spawn_is_expanded_10mm(env).bool().detach().cpu()
                )

            egg_xy_local = egg.data.root_link_pos_w[:, :2] - origins
            min_robot_clearance = min(
                min_robot_clearance,
                torch.abs(egg_xy_local[:, 0]).min().item(),
            )
            bucket_xy = BUCKET_XY.to(device=env.device)
            min_bucket_clearance = min(
                min_bucket_clearance,
                torch.linalg.norm(egg_xy_local - bucket_xy, dim=-1).min().item(),
            )

            require(not torch.any(mdp.egg_fell(env)).item(), "egg_fell at reset")
            require(
                not torch.any(mdp.egg_out_of_bounds(env)).item(),
                "egg_oob at reset",
            )
            require(not torch.any(mdp.nan_state(env)).item(), "NaN at reset")
    finally:
        env.close()

    offsets = torch.cat(all_offsets)
    nominal = torch.cat(all_nominal)
    strata = torch.cat(all_strata)
    rejections = torch.cat(all_rejections)
    randomized_offsets = offsets[~nominal]
    observed_strata = torch.unique(strata[strata >= 0])
    nominal_fraction = nominal.to(torch.float32).mean().item()

    require(max_pair_error <= 1.0e-6, "egg and pedestal offsets differ")
    require(
        torch.max(torch.abs(offsets)).item() <= range_m + 1.0e-6,
        f"spawn outside configured +/-{range_m * 1000.0:g} mm range",
    )
    require(len(randomized_offsets) > 0, "no randomized reset was produced")
    require(len(observed_strata) == 25, "not all 25 spatial strata were sampled")
    require(
        torch.max(rejections).item() == 0,
        f"+/-{range_m * 1000.0:g} mm should not require rejection sampling",
    )
    if profile["kind"] == "single_range":
        require(
            abs(nominal_fraction - 1.0 / nominal_every_n) <= 1.0 / args.resets,
            (
                "nominal reset fraction is inconsistent with "
                f"one-in-{nominal_every_n} schedule"
            ),
        )
        group_report = None
    else:
        core = torch.cat(all_core)
        expanded = torch.cat(all_expanded)
        membership_count = (
            nominal.to(torch.int8)
            + core.to(torch.int8)
            + expanded.to(torch.int8)
        )
        require(
            torch.all(membership_count == 1).item(),
            "each S2-C reset must belong to exactly one spawn group",
        )
        fractions = torch.stack(
            [
                nominal.to(torch.float32).mean(),
                core.to(torch.float32).mean(),
                expanded.to(torch.float32).mean(),
            ]
        )
        expected = torch.tensor(profile["group_fractions"])
        fraction_tolerance = 1.0 / args.num_envs + 1.0 / len(offsets)
        require(
            torch.max(torch.abs(fractions - expected)).item()
            <= fraction_tolerance,
            "S2-C group fractions do not match the 10/30/60 schedule",
        )
        core_offsets = offsets[core]
        expanded_offsets = offsets[expanded]
        require(len(core_offsets) > 0, "no S2-C core reset was produced")
        require(len(expanded_offsets) > 0, "no S2-C expanded reset was produced")
        require(
            torch.max(torch.abs(core_offsets)).item()
            <= float(profile["core_range_m"]) + 1.0e-6,
            "S2-C core spawn exceeded +/-5 mm",
        )
        require(
            torch.any(
                torch.max(torch.abs(expanded_offsets), dim=1).values
                > float(profile["core_range_m"])
            ).item(),
            "S2-C expanded group did not sample outside the +/-5 mm core",
        )
        core_strata = torch.unique(strata[core & (strata >= 0)])
        expanded_strata = torch.unique(strata[expanded & (strata >= 0)])
        require(len(core_strata) == 25, "S2-C core missed spatial strata")
        require(len(expanded_strata) == 25, "S2-C expanded group missed strata")
        group_report = fractions.tolist()
    require(
        min_robot_clearance >= ROBOT_CLEARANCE_M - 1.0e-6,
        "spawn failed robot clearance rule",
    )
    require(
        min_bucket_clearance >= BUCKET_CLEARANCE_M - 1.0e-6,
        "spawn failed bucket clearance rule",
    )

    print("Stage-2 spawn check: PASS")
    print(f"  task:                     {args.task_id}")
    print(f"  configured half-range:    {range_m * 1000.0:.1f} mm")
    print(f"  environments:             {args.num_envs}")
    print(f"  resets checked:           {args.resets}")
    print(f"  initial states checked:   {len(offsets)}")
    print(f"  nominal fraction:         {nominal_fraction:.3f}")
    if group_report is not None:
        print(f"  core +/-5 mm fraction:    {group_report[1]:.3f}")
        print(f"  expanded +/-10 mm frac.:  {group_report[2]:.3f}")
    print(f"  spatial strata observed:  {len(observed_strata)}/25")
    print(f"  rejected candidates:      {rejections.sum().item()}")
    print(f"  maximum pair error:       {max_pair_error * 1000.0:.6f} mm")
    print(f"  minimum |egg x|:          {min_robot_clearance * 1000.0:.3f} mm")
    print(
        "  minimum egg-bin distance: "
        f"{min_bucket_clearance * 1000.0:.3f} mm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
