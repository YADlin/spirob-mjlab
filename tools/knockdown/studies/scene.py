"""Plain-MuJoCo assembly of the Lab robot + spirob-mjlab egg/pedestal (CPU studies)."""
import mujoco, numpy as np
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[3]
LAB=str(_ROOT/"src/spirob_mjlab/assets/lab_model/spirob_robot.xml")
A=str(_ROOT/"src/spirob_mjlab/assets")+"/"
C_ORIGIN_Y=0.21796009  # link_002 rest y (Lab curve-frame origin)
def c_to_world(u,v): return np.array([u, C_ORIGIN_Y - v])
def build(egg_uv=(0.05,0.10), dt=1e-3, robot=LAB):
    s=mujoco.MjSpec.from_file(robot)
    s.option.timestep=dt; s.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    xy=c_to_world(*egg_uv)
    for name,fn,z in (("ped",A+"spirob_pedestal.xml",0.002),("egg",A+"spirob_egg.xml",0.098)):
        c=mujoco.MjSpec.from_file(fn)
        fr=s.worldbody.add_frame(pos=[xy[0],xy[1],z])
        fr.attach_body(c.worldbody.first_body(), name+"/", "")
    m=s.compile(); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
    d.ctrl[:]=d.ten_length[:2]  # neutral payout: actuators hold the rest shape
    return m,d
