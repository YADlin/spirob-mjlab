"""Replay a frozen actor using real MuJoCo RGB and the Lab's black-egg detector.

No hardware I/O. Video renders intermediate physics states at 30 fps while the
actor still decides every 1.2 s. This is a nominal CPU demonstration, not the
randomized 200-episode acceptance evaluation (use evaluate.py for that).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from spirob_mjlab.knockdown import contract as C, mdp
from spirob_mjlab.knockdown.camera import CAMERA_NAME, WIDTH, HEIGHT, ImageTracker
from spirob_mjlab.knockdown.checkpoint import load_actor, action
from spirob_mjlab.knockdown.entities import SOURCE_XML, sha256
from spirob_mjlab.knockdown.plain_scene import build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--egg-u", type=float, default=0.08)
    p.add_argument("--egg-v", type=float, default=0.11)
    p.add_argument("--start-diff-mm", type=float, default=0.0)
    p.add_argument("--common-speed-mm-s", type=float, default=5.0)
    p.add_argument("--video", action="store_true")
    p.add_argument("--hold", action="store_true", help="Hold initial D; do not run the actor")
    a = p.parse_args()
    if not 5 <= a.common_speed_mm_s <= 20 or not -80 <= a.start_diff_mm <= 80:
        p.error("Common speed must be 5--20 mm/s and initial D within +/-80 mm")
    uv = (a.egg_u, a.egg_v)
    table = C.load_pose_table(mdp.POSE_TABLE)
    row = min(table["rows"], key=lambda r: abs(r["diff_mm"] - a.start_diff_mm))
    diff = row["diff_mm"] / 1000
    if C.robot_clearance(uv, diff, table) <= C.DEFAULT.autostart.clearance_m:
        p.error("Requested placement/start D violates clearance; choose a different pair")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    actor = load_actor(a.checkpoint)
    model, data = build(egg_uv=uv, dt=.002, diff_m=diff, qpos=row["qpos"], camera=True)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    tracker = ImageTracker()
    egg_body = model.body("egg/egg").id
    ped_body = model.body("ped/pedestal").id
    floor = model.geom("floor").id
    touched = floor_touched = False
    previous = diff / C.DEFAULT.action.diff_limit_m
    speed = a.common_speed_mm_s / 1000
    records = []
    writer = None
    if a.video:
        # Stream frames; don't retain a whole episode in memory.
        import imageio.v2 as imageio
        writer = imageio.get_writer(out / "camera.mp4", fps=30, macro_block_size=1)
    next_frame = 0.0
    outcome = "time_out"

    def render():
        renderer.update_scene(data, camera=CAMERA_NAME)
        return renderer.render()

    def frame():
        rgb = render()
        overlay = rgb.copy()
        cv2.putText(overlay, f"t={data.time:.2f}s  D={diff*1000:+.1f}mm  speed={a.common_speed_mm_s:g}mm/s",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
        display_detection = tracker.detector.detect(rgb)
        if display_detection is not None:
            px = tuple(int(round(x)) for x in display_detection.px)
            cv2.circle(overlay, px, 8, (0, 255, 0), 1)
        if writer is not None:
            writer.append_data(overlay)
        return overlay

    try:
        for decision in range(C.DEFAULT.episode_decisions):
            observed, visible, start = tracker.update(render())
            if decision == 0:
                cv2.imwrite(str(out / "initial.png"), frame()[:, :, ::-1])
                if not visible:
                    outcome = "invalid_perception_reset"
                    break
            obs = C.build_actor_obs(observed, np.array([visible]), start,
                                    np.array([diff]), np.array([previous]))
            raw = previous if a.hold else float(action(actor, obs)[0, 0])
            clipped = float(C.clip_raw(np.asarray(raw)))
            target = float(C.limited_diff(C.desired_diff(np.asarray(clipped)), np.asarray(diff)))
            records.append(dict(decision=decision, time_s=data.time, visible=visible,
                                egg_u=observed[0], egg_v=observed[1], start_u=start[0], start_v=start[1],
                                encoder_D_mm=diff*1000, previous_action=previous, raw_action=raw,
                                target_D_mm=target*1000, common_speed_mm_s=a.common_speed_mm_s))
            previous = clipped
            for _ in range(round(C.DEFAULT.action.decision_period_s / model.opt.timestep)):
                diff += float(np.clip(target - diff, -2*speed*model.opt.timestep, 2*speed*model.opt.timestep))
                data.ctrl[:] = C.payouts_from_diff(diff)
                mujoco.mj_step(model, data)
                if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.qacc).all()):
                    outcome = "numerical_fault"
                    break
                for contact in data.contact[:data.ncon]:
                    bodies = (model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2])
                    if egg_body in bodies:
                        other = bodies[1] if bodies[0] == egg_body else bodies[0]
                        touched |= other not in (0, egg_body, ped_body)
                        floor_touched |= floor in (contact.geom1, contact.geom2)
                if writer is not None and data.time + 1e-9 >= next_frame:
                    frame()
                    next_frame += 1/30
                if floor_touched:
                    outcome = "knocked_down" if touched else "spontaneous_fall"
                    break
                if (np.abs(data.xpos[egg_body, :2]) > .4).any():
                    outcome = "egg_out_of_bounds"
                    break
            if outcome != "time_out":
                break
        cv2.imwrite(str(out / "final.png"), frame()[:, :, ::-1])
    finally:
        renderer.close()
        if writer is not None:
            writer.close()
    if records:
        with (out / "decisions.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    result = dict(outcome=outcome, decisions=len(records), simulated_seconds=float(data.time),
                  initial_diff_mm=row["diff_mm"], egg_uv=uv, common_speed_mm_s=a.common_speed_mm_s,
                  robot_egg_contact=bool(touched), egg_floor_contact=bool(floor_touched),
                  contract_id=C.CONTRACT_ID, perception="MuJoCo OpenGL RGB -> shared Lab detector",
                  nominal_dynamics=True, checkpoint_sha256=sha256(Path(a.checkpoint)),
                  source_model_sha256=sha256(SOURCE_XML), hold_baseline=a.hold,
                  detector_sha256=sha256(Path(__file__).parents[2] / "src/spirob_mjlab/knockdown/egg_vision.py"))
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
