"""Plain-MuJoCo (CPU) assembly of the knockdown scene (same files as the mjlab task)."""

from __future__ import annotations

import mujoco
import numpy as np

from spirob_mjlab.knockdown import contract as C
from spirob_mjlab.knockdown.entities import EGG_XML, PEDESTAL_XML, robot_spec
from spirob_mjlab.knockdown.mdp import EGG_SETTLED_Z, PEDESTAL_ROOT_Z


def build(egg_uv=(0.08, 0.11), dt=1e-3, diff_m=0.0, ref_offset_m=(0.0, 0.0), robot_dz=0.0, qpos=None,
          camera=False):
    s = robot_spec()
    floor = s.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05])
    if camera:
        from spirob_mjlab.knockdown.camera import add_camera, FLOOR_RGBA
        add_camera(s)
        for geom in s.geoms:
            geom.rgba = (0.9, 0.9, 0.9, 1.0)
        floor.rgba = FLOOR_RGBA
        s.worldbody.add_light(name="sun", pos=(0, 0, 1.5), dir=(0, 0, -1),
                              type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    s.option.timestep = dt
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if robot_dz:
        b = s.body("robot_base")
        b.pos = [b.pos[0], b.pos[1], b.pos[2] + robot_dz]
    x, y = C.c_to_model_xy(*egg_uv)
    for prefix, path, z in (("ped/", PEDESTAL_XML, PEDESTAL_ROOT_Z), ("egg/", EGG_XML, EGG_SETTLED_Z)):
        child = mujoco.MjSpec.from_file(str(path))
        frame = s.worldbody.add_frame(pos=[x, y, z])
        frame.attach_body(child.worldbody.first_body(), prefix, "")
    m = s.compile()
    d = mujoco.MjData(m)
    if qpos is not None:
        for i, q in enumerate(qpos):
            d.qpos[m.jnt_qposadr[m.joint(f"j_{i+1:03d}").id]] = q
    d.ctrl[:] = np.asarray(C.payouts_from_diff(diff_m)) + np.asarray(ref_offset_m)
    mujoco.mj_forward(m, d)
    return m, d


def robot_egg_contacts(m, d) -> int:
    egg, ped = m.body("egg/egg").id, m.body("ped/pedestal").id
    robot = {i for i in range(m.nbody) if i not in (0, egg, ped)}
    return sum(1 for c in d.contact[: d.ncon]
               if (m.geom_bodyid[c.geom1] in robot and m.geom_bodyid[c.geom2] in (egg, ped))
               or (m.geom_bodyid[c.geom2] in robot and m.geom_bodyid[c.geom1] in (egg, ped)))
