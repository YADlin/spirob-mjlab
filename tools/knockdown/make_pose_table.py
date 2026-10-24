"""Quasi-static robot shapes over the cable-difference range (knockdown model, CPU MuJoCo).

For D = L0 - L1 from -80 to +80 mm (step 2.5 mm) at the fixed mean payout, ramp D slowly from 0,
hold, and record: joint angles, link centre XY in frame C, link half-widths and the tendon lengths.
Used by (1) the training reset (random start shape) and (2) the Lab's clearance check before an
automatic episode start. Output: assets/knockdown/pose_table.json
"""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

R = Path(__file__).resolve().parents[2]
XML = R / "src/spirob_mjlab/assets/knockdown/spirob_robot.xml"
OUT = R / "src/spirob_mjlab/assets/knockdown/pose_table.json"
STEP_MM, MAX_MM, RATE = 2.5, 80.0, 0.005  # m/s of D (2.5 mm/s per cable)


def halfwidths(m, d):
    hw = []
    for i in range(1, 22):
        bid = m.body(f"link_{i:03d}").id
        vs = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == bid and m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                vs.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        loc = (np.concatenate(vs) - d.xpos[bid]) @ d.xmat[bid].reshape(3, 3)
        hw.append(float(np.abs(loc[:, 0]).max()))
    return hw


def main():
    m = mujoco.MjModel.from_xml_path(str(XML))
    m.opt.timestep = 1e-3
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    neutral = d.ten_length[:2].copy()
    mean = float(neutral.mean())
    origin = d.body("link_002").xpos[:2].copy()
    hw = halfwidths(m, d)
    links = [m.body(f"link_{i:03d}").id for i in range(1, 22)]
    jadr = [m.jnt_qposadr[m.joint(f"j_{i:03d}").id] for i in range(1, 22)]
    grid = np.round(np.arange(-MAX_MM, MAX_MM + 1e-9, STEP_MM), 3)
    rows = {}
    for sign in (+1, -1):
        mujoco.mj_resetData(m, d)
        D = 0.0
        d.ctrl[:] = neutral
        for _ in range(500):
            mujoco.mj_step(m, d)
        targets = [g for g in grid if (g >= 0 if sign > 0 else g <= 0)]
        targets = sorted(targets, key=abs)
        for g in targets:
            goal = g / 1000
            while abs(goal - D) > 1e-9:
                D += np.clip(goal - D, -RATE * m.opt.timestep, RATE * m.opt.timestep)
                d.ctrl[:] = [mean + D / 2, mean - D / 2]
                mujoco.mj_step(m, d)
            for _ in range(800):  # 0.8 s hold
                mujoco.mj_step(m, d)
            xy = d.xpos[links][:, :2]
            uv = np.stack([xy[:, 0] - origin[0], origin[1] - xy[:, 1]], 1)
            tip = d.site("tip_site").xpos[:2]
            rows[float(g)] = dict(
                diff_mm=float(g), qpos=[float(d.qpos[a]) for a in jadr],
                link_uv=uv.round(6).tolist(), tip_uv=[float(tip[0] - origin[0]), float(origin[1] - tip[1])],
                tendon_m=d.ten_length[:2].round(7).tolist(), qvel_max=float(np.abs(d.qvel).max()))
    table = dict(model_sha256=hashlib.sha256(XML.read_bytes()).hexdigest(), mean_payout_m=mean,
                 neutral_m=neutral.tolist(), c_origin_model_xy=origin.tolist(), link_halfwidth_m=hw,
                 step_mm=STEP_MM, rows=[rows[k] for k in sorted(rows)])
    OUT.write_text(json.dumps(table))
    tips = {int(k): np.round(np.array(rows[k]["tip_uv"]) * 100, 1).tolist() for k in (-80, -40, 0, 40, 80)}
    print("tip (u,v) cm at D mm:", tips, "| max residual qvel", max(r["qvel_max"] for r in rows.values()))


if __name__ == "__main__":
    main()
