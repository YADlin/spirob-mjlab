"""Check the calibrated camera in the real mjlab RGB pipeline. No policy or hardware.

Writes one sensor preview and its measured detector errors. Tests two separated
environment origins at reset and after one held-D decision. This is a sensing
regression only, not a trained-policy success test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import spirob_mjlab  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from spirob_mjlab.knockdown import mdp
from spirob_mjlab.knockdown.camera import ImageTracker, camera_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_env_cfg("Mjlab-SpiRob-EggKnockdown-Camera", play=True)
    cfg.seed = 23
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device=args.device)
    measurements = []
    try:
        for step in range(2):
            if step == 0:
                obs, _ = env.reset()
            else:
                hold = env.action_manager.get_term(mdp.ACTION_TERM).diff / mdp.C.DEFAULT.action.diff_limit_m
                obs, _, _, _, _ = env.step(hold.clone())
            truth = mdp.egg_uv_true(env).cpu().numpy()
            frames = env.scene["overhead_rgb"].data.rgb.cpu().numpy()
            for i, rgb in enumerate(frames):
                uv, visible, _ = ImageTracker().update(rgb)
                error = float(np.linalg.norm(uv - truth[i]))
                assert visible == 1, (step, i, "egg not visible")
                assert error < .003, (step, i, error)
                assert np.allclose(obs["actor"][i, :2].cpu().numpy() * .1, uv, atol=1e-6)
                measurements.append(dict(step=step, environment=i, uv_detected_m=uv.tolist(),
                                         uv_true_m=truth[i].tolist(), error_mm=error*1000))
            if step == 0:
                cv2.imwrite(str(out / "sensor-preview.png"), frames[0][:, :, ::-1])
        result = dict(device=args.device, environments=env.num_envs,
                      origins_m=env.scene.env_origins.cpu().numpy().tolist(),
                      rendered_camera=camera_manifest(), measurements=measurements,
                      max_detection_error_mm=max(r["error_mm"] for r in measurements),
                      scope="Four image detections; no policy evaluation, training or hardware")
        (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "rendered_camera"}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
