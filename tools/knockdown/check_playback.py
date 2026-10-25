"""Headless regression of the real native-viewer path with random actions.

Replaces only the desktop window with an EGL recorder. Runs the rendered-camera
environment, then repeats the same actions without a viewer and compares states.
This is a playback check, not training or a success-rate evaluation.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer import NativeMujocoViewer
from spirob_mjlab.knockdown.playback import PhysicsFramePlayback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=4)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    actions = 2 * torch.rand(a.steps, 1, 1, generator=torch.Generator().manual_seed(101)) - 1
    frames, viewed, plain = [], [], []

    def make_env():
        cfg = load_env_cfg("Mjlab-SpiRob-EggKnockdown-Camera", play=True)
        cfg.seed = 17
        cfg.scene.num_envs = 1
        return RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"))

    def snapshot(env):
        base = env.unwrapped
        return np.concatenate([
            getattr(base.sim.data, key).cpu().numpy().ravel().copy()
            for key in ("qpos", "qvel", "ctrl")
        ] + [env.get_observations()["actor"].cpu().numpy().ravel().copy(),
             base.reward_buf.cpu().numpy().copy(), base.reset_buf.cpu().numpy().copy()])

    env = make_env()
    policy_calls = 0

    def policy(obs):
        nonlocal policy_calls
        del obs
        action = actions[policy_calls]
        policy_calls += 1
        return action

    writer = imageio.get_writer(out / "random-camera-playback.mp4", fps=60, macro_block_size=1)

    class Window:
        def __init__(self, model, data, **kwargs):
            self.data = data
            self.renderer = mujoco.Renderer(model, height=360, width=640)
            self.user_scn = mujoco.MjvScene(model, maxgeom=10000)
            self.cam, self.opt, self.perturb = mujoco.MjvCamera(), mujoco.MjvOption(), mujoco.MjvPerturb()
            self.open = True
            self.last_step = -1

        def is_running(self): return self.open
        def set_texts(self, *args): pass
        def set_figures(self, *args): pass

        def sync(self, **kwargs):
            base = env.unwrapped
            step = base._sim_step_counter
            if step == self.last_step:
                return
            self.last_step = step
            self.renderer.update_scene(self.data, camera="camera/overhead")
            writer.append_data(self.renderer.render())
            act = base.action_manager.get_term("cable_diff")
            frames.append(dict(substep=step, simulation_seconds=step*base.physics_dt,
                               D_mm=float(act.diff[0, 0])*1000, target_D_mm=float(act.limited[0, 0])*1000))

        def close(self):
            self.open = False
            self.renderer.close()

    class Viewer(PhysicsFramePlayback, NativeMujocoViewer):
        def setup(self):
            super().setup()
            self.sync_env_to_viewer()

        def _execute_step(self):
            ok = super()._execute_step()
            assert ok, self._last_error
            viewed.append(snapshot(env))
            return ok

    try:
        with patch("mujoco.viewer.launch_passive", Window):
            viewer = Viewer(env, policy)
            viewer.run(num_steps=a.steps)
    finally:
        writer.close()
        env.close()

    env = make_env()
    try:
        for action in actions:
            env.step(action)
            plain.append(snapshot(env))
    finally:
        env.close()
    error = float(np.max(np.abs(np.asarray(viewed) - np.asarray(plain))))
    result = dict(task="Mjlab-SpiRob-EggKnockdown-Camera", agent="random", device="cpu",
                  physics_steps=a.steps*600, policy_calls=policy_calls, frames=len(frames),
                  frames_per_decision=72, frame_rate=60, decision_period_s=1.2,
                  max_state_observation_reward_error=error,
                  desktop_window="replaced with EGL recorder; native sync methods unchanged",
                  actions=actions.flatten().tolist())
    (out / "frames.json").write_text(json.dumps(frames, indent=2))
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert policy_calls == a.steps
    assert len(frames) == 1 + 72*a.steps
    assert error < 1e-7


if __name__ == "__main__":
    main()
