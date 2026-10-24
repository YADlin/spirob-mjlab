"""Plain-MuJoCo (CPU) assembly of the knockdown scene for tests and studies.

Same files as the mjlab task: Lab robot (floor removed by the entity spec_fn,
re-added here as a plane), spirob-mjlab egg and pedestal placed in frame C.
"""

from __future__ import annotations

import mujoco
import numpy as np

from spirob_mjlab.entities import SPIROB_EGG_XML, SPIROB_PEDESTAL_XML
from spirob_mjlab.knockdown import contract as C
from spirob_mjlab.knockdown.entities import lab_robot_spec
from spirob_mjlab.knockdown.mdp import EGG_SETTLED_Z, PEDESTAL_ROOT_Z


def build(egg_uv=(0.075, 0.115), dt=1e-3, ref_offset_m=(0.0, 0.0)):
    s = lab_robot_spec()
    s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05])
    s.option.timestep = dt
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    x, y = C.c_to_model_xy(*egg_uv)
    for prefix, path, z in (("ped/", SPIROB_PEDESTAL_XML, PEDESTAL_ROOT_Z), ("egg/", SPIROB_EGG_XML, EGG_SETTLED_Z)):
        child = mujoco.MjSpec.from_file(str(path))
        frame = s.worldbody.add_frame(pos=[x, y, z])
        frame.attach_body(child.worldbody.first_body(), prefix, "")
    m = s.compile()
    d = mujoco.MjData(m)
    d.ctrl[:] = np.asarray(C.NEUTRAL_PAYOUT_M) + np.asarray(ref_offset_m)
    mujoco.mj_forward(m, d)
    return m, d


def robot_egg_contacts(m, d) -> int:
    egg = m.body("egg/egg").id
    ped = m.body("ped/pedestal").id
    robot = {i for i in range(m.nbody) if i not in (0, egg, ped)}
    n = 0
    for c in d.contact[: d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 in robot and b2 in (egg, ped)) or (b2 in robot and b1 in (egg, ped)):
            n += 1
    return n
