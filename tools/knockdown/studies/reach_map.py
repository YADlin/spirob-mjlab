"""Scripted-curl knockdown map over egg placements (Lab model, CPU MuJoCo)."""
import sys, json, numpy as np, mujoco, itertools
from multiprocessing import Pool
from pathlib import Path; sys.path.insert(0,str(Path(__file__).resolve().parent)); from scene import build
SPEED=0.020; SHORTEN=0.070; HOLD=1.5
def trial(args):
    u,v,cable=args
    m,d=build(egg_uv=(u,v),dt=1e-3)
    egg=m.body('egg/egg').id; ped=m.body('ped/pedestal').id
    robot_geoms={i for i in range(m.ngeom) if m.geom_bodyid[i] not in (0,egg,ped)}
    egg_geoms={i for i in range(m.ngeom) if m.geom_bodyid[i]==egg}
    if any((c.geom1 in robot_geoms and c.geom2 in egg_geoms) or (c.geom2 in robot_geoms and c.geom1 in egg_geoms) for c in d.contact[:d.ncon]):
        return dict(u=u,v=v,cable=cable,status='invalid_overlap')
    neutral=d.ctrl.copy()
    for _ in range(300): mujoco.mj_step(m,d)  # settle 0.3 s
    z0=d.xpos[egg,2]; p0=d.xpos[egg,:2].copy(); touched=False
    T=SHORTEN/SPEED+HOLD; n=int(T/1e-3)
    for k in range(n):
        t=k*1e-3; ctrl=neutral.copy(); x=min(SHORTEN, SPEED*t); ctrl[cable]-=x; ctrl[1-cable]+=x; d.ctrl[:]=ctrl
        mujoco.mj_step(m,d)
        if not touched:
            touched=any((c.geom1 in robot_geoms and c.geom2 in egg_geoms) or (c.geom2 in robot_geoms and c.geom1 in egg_geoms) for c in d.contact[:d.ncon])
    if not np.isfinite(d.qacc).all(): return dict(u=u,v=v,cable=cable,status='nan')
    z=d.xpos[egg,2]; disp=float(np.linalg.norm(d.xpos[egg,:2]-p0))
    fell = z < z0-0.02
    return dict(u=u,v=v,cable=cable,status='knocked' if fell else ('touched' if touched else 'missed'),dz_mm=float((z-z0)*1e3),disp_mm=disp*1e3)
if __name__=='__main__':
    us=np.round(np.arange(-0.20,0.2001,0.02),3); vs=np.round(np.arange(0.0,0.2401,0.02),3)
    jobs=[(float(u),float(v),c) for u,v,c in itertools.product(us,vs,(0,1))]
    with Pool(2) as p: res=p.map(trial,jobs,chunksize=4)
    json.dump(res,open(str(Path(__file__).resolve().parents[3]/'docs/knockdown/v1/reach_map.json'),'w'))
    from collections import Counter; print(Counter(r['status'] for r in res))
