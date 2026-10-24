"""Compare Lab-model motion across physics timesteps/integrators (CPU MuJoCo 3.8.1)."""
import mujoco, numpy as np, time, json
from pathlib import Path
P=str(Path(__file__).resolve().parents[3]/"src/spirob_mjlab/assets/lab_model/spirob_robot.xml")
def run(dt, integ, T=3.0, speed=0.020):
    m=mujoco.MjModel.from_xml_path(P); m.opt.timestep=dt
    m.opt.integrator={'implicitfast':mujoco.mjtIntegrator.mjINT_IMPLICITFAST,'implicit':mujoco.mjtIntegrator.mjINT_IMPLICIT,'euler':mujoco.mjtIntegrator.mjINT_EULER}[integ]
    d=mujoco.MjData(m); mujoco.mj_forward(m,d)
    L0=d.ten_length.copy(); tgt=L0+np.array([-0.030,0.030])
    tip=m.site('tip_site').id; out=[]; n=int(T/dt); t0=time.perf_counter(); ok=True
    for k in range(n):
        t=k*dt; s=min(1.0, t*speed/0.030)
        d.ctrl[:]=L0+(tgt-L0)*s
        mujoco.mj_step(m,d)
        if k % int(0.01/dt)==0: out.append(d.site_xpos[tip,:2].copy())
        if not np.isfinite(d.qacc).all(): ok=False; break
    wall=time.perf_counter()-t0
    return np.array(out), ok, wall/T, d.ten_length.copy()
ref,_,w,Lr=run(1e-4,'implicitfast')
res={'ref_wall_per_simsec':w,'ref_final_tip_mm':(ref[-1]*1e3).round(2).tolist(),'ref_final_ten_mm':(Lr*1e3).round(3).tolist()}
for integ in ['implicitfast','euler']:
  for dt in [1e-4,2e-4,5e-4,1e-3,2e-3]:
    tr,ok,w,L=run(dt,integ)
    n=min(len(tr),len(ref)); err=np.abs(tr[:n]-ref[:n]).max()*1e3 if ok else float('nan')
    res[f'{integ}_{dt}']=dict(finite=ok, max_tip_dev_mm=round(float(err),3), wall_per_simsec=round(w,3), final_tip_mm=(tr[-1]*1e3).round(2).tolist())
    print(integ,dt,res[f'{integ}_{dt}'],flush=True)
print(json.dumps({k:v for k,v in res.items() if k.startswith('ref')}))
json.dump(res,open(str(Path(__file__).resolve().parents[3]/'docs/knockdown/timestep_study.json'),'w'),indent=1)
