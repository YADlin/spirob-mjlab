import sys, json, itertools, numpy as np, mujoco
from multiprocessing import Pool
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from spirob_mjlab.knockdown.plain_scene import build as _b
def build(egg_uv, robot_dz=0.0): return _b(egg_uv=egg_uv, robot_dz=robot_dz)
V_CABLE=0.005; DMAX=0.080
def trial(a):
    u,v,sign,dz=a
    m,d=build(egg_uv=(u,v),robot_dz=dz); egg=m.body('egg/egg').id; ped=m.body('ped/pedestal').id
    rg={g for g in range(m.ngeom) if m.geom_bodyid[g] not in (0,egg,ped)}; eg={g for g in range(m.ngeom) if m.geom_bodyid[g] in (egg,ped)}
    hit=lambda: any((c.geom1 in rg and c.geom2 in eg) or (c.geom2 in rg and c.geom1 in eg) for c in d.contact[:d.ncon])
    if hit(): return dict(u=u,v=v,sign=sign,dz=dz,status='invalid_overlap')
    n0=d.ctrl.copy(); mean=n0.mean(); z0=d.xpos[egg,2]; touched=False; tfall=None; pedhit=False
    T=DMAX/(2*V_CABLE)+1.5; n=int(T/1e-3)
    pg={g for g in range(m.ngeom) if m.geom_bodyid[g]==ped}
    for k in range(n):
        D=sign*min(DMAX,2*V_CABLE*k*1e-3); d.ctrl[:]=[mean+D/2,mean-D/2]; mujoco.mj_step(m,d)
        if k%20==0:
            for c in d.contact[:d.ncon]:
                if (c.geom1 in rg and c.geom2 in eg) or (c.geom2 in rg and c.geom1 in eg):
                    touched=True
                    if c.geom1 in pg or c.geom2 in pg: pedhit=True
        if tfall is None and d.xpos[egg,2]<0.046-0.005: tfall=k*1e-3
    st='knocked' if tfall is not None else ('touched' if touched else 'missed')
    return dict(u=u,v=v,sign=sign,dz=dz,status=st,t_fall=tfall,D_at_fall_mm=None if tfall is None else round(min(80,10*tfall)*sign,1),ped_contact=pedhit)
if __name__=='__main__':
    us=np.round(np.arange(-0.20,0.2001,0.02),3); vs=np.round(np.arange(-0.04,0.2401,0.02),3)
    jobs=[(float(u),float(v),s,dz) for dz in (0.0,-0.012) for u,v,s in itertools.product(us,vs,(1,-1))]
    with Pool(2) as p: res=p.map(trial,jobs,chunksize=6)
    json.dump(res,open(str(Path(__file__).resolve().parents[3]/'docs/knockdown/reach_map_v2.json'),'w')); from collections import Counter
    for dz in (0.0,-0.012): print(dz, Counter(r['status'] for r in res if r['dz']==dz), 'pedestal contacts', sum(r.get('ped_contact',False) for r in res if r['dz']==dz))
