"""CPU/GPU smoke test for the knockdown task: build, reset, step, classify.

usage: uv run python tools/knockdown/smoke_cpu.py [--device cpu|cuda:0] [--envs 2] [--steps 3] [--task ...]
"""
import argparse, time, json, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import spirob_mjlab  # noqa: F401  (registers tasks)
from spirob_mjlab.knockdown import mdp, contract as C

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu"); p.add_argument("--envs", type=int, default=2)
p.add_argument("--steps", type=int, default=3); p.add_argument("--task", default="Mjlab-SpiRob-EggKnockdown-Fixed")
p.add_argument("--policy", default="zero", choices=["zero", "curl_pos", "curl_neg", "random"])
a = p.parse_args()
cfg = load_env_cfg(a.task); cfg.scene.num_envs = a.envs
t = time.time(); env = ManagerBasedRlEnv(cfg, device=a.device); print("build_s", round(time.time() - t, 1))
obs, _ = env.reset()
print("obs", {k: tuple(v.shape) for k, v in obs.items()}, "actor_dim(contract)", C.DEFAULT.actor_dim)
egg0 = mdp.egg_pos_w(env).clone(); uv0 = mdp.egg_uv_true(env).clone()
print("egg_uv_true", uv0.tolist()); print("actor", obs["actor"][0].tolist())
act = {"zero": torch.zeros(a.envs, 2), "curl_pos": torch.tensor([[-1.0, 1.0]]).repeat(a.envs, 1),
       "curl_neg": torch.tensor([[1.0, -1.0]]).repeat(a.envs, 1)}
for k in range(a.steps):
    u = act.get(a.policy, None)
    u = (2 * torch.rand(a.envs, 2) - 1) if u is None else u
    t = time.time(); obs, r, term, trunc, info = env.step(u.to(env.device))
    at = env.action_manager.get_term(mdp.ACTION_TERM)
    tm = env.termination_manager
    fired = {n: tm.get_term(n).nonzero().flatten().tolist() for n in tm.active_terms if tm.get_term(n).any()}
    print(f"step {k} wall {time.time()-t:.1f}s r={r.tolist()} payout_mm={(at.payout*1e3).round().tolist()} "
          f"egg_dz_mm={((mdp.egg_pos_w(env)[:,2]-egg0[:,2])*1e3).tolist()} touched={mdp.egg_touched(env).tolist()} term={fired}")
