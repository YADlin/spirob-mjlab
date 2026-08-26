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

TASK_ID = "Mjlab-SpiRob-EggToBucket-Stage2"
RANGE_M = 0.002
ROBOT_CLEARANCE_M = 0.038
BUCKET_CLEARANCE_M = 0.055
BUCKET_XY = torch.tensor((-0.05, 0.15))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--resets", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    args = parse_args()
    require(args.num_envs >= 25, "Use at least 25 environments to cover all strata")
    require(args.resets >= 4, "Use at least four resets to check nominal retention")

    cfg = load_env_cfg(TASK_ID)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)

    all_offsets = []
    all_nominal = []
    all_strata = []
    all_rejections = []
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
        torch.max(torch.abs(offsets)).item() <= RANGE_M + 1.0e-6,
        "spawn outside configured Â±2 mm range",
    )
    require(len(randomized_offsets) > 0, "no randomized reset was produced")
    require(len(observed_strata) == 25, "not all 25 spatial strata were sampled")
    require(
        torch.max(rejections).item() == 0,
        "Â±2 mm should not require clearance-screen rejection sampling",
    )
    require(
        abs(nominal_fraction - 0.25) <= 1.0 / args.resets,
        "nominal reset fraction is inconsistent with one-in-four schedule",
    )
    require(
        min_robot_clearance >= ROBOT_CLEARANCE_M - 1.0e-6,
        "spawn failed robot clearance rule",
    )
    require(
        min_bucket_clearance >= BUCKET_CLEARANCE_M - 1.0e-6,
        "spawn failed bucket clearance rule",
    )

    print("Stage-2 spawn check: PASS")
    print(f"  environments:             {args.num_envs}")
    print(f"  resets checked:           {args.resets}")
    print(f"  initial states checked:   {len(offsets)}")
    print(f"  nominal fraction:         {nominal_fraction:.3f}")
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
