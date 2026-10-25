"""Playback must expose real substeps without changing the policy/control clock."""
from types import SimpleNamespace
from collections import deque

import pytest

from spirob_mjlab.knockdown.playback import PhysicsFramePlayback, physics_frames


class Sim:
    def __init__(self):
        self.position = 0

    def step(self):
        self.position += 1


def test_one_decision_shows_integrated_motion_and_restores_sim():
    sim = Sim()
    seen = []
    with physics_frames(sim, .002, 60, lambda t, dt: seen.append((t, sim.position))):
        for _ in range(600):
            sim.step()
    assert sim.position == 600
    assert len(seen) == 72
    assert seen[0] == pytest.approx((.018, 9))
    assert seen[-1] == (1.2, 600)
    assert "step" not in vars(sim)


def test_callback_failure_restores_existing_override():
    calls = []
    original = lambda: calls.append(1)
    sim = SimpleNamespace(step=original)

    def fail(*args):
        raise RuntimeError("window failed")

    with pytest.raises(RuntimeError, match="window failed"):
        with physics_frames(sim, .002, 60, fail):
            for _ in range(600):
                sim.step()
    assert sim.step is original
    assert len(calls) == 9


def test_viewer_paces_one_policy_decision_without_extra_wait(monkeypatch):
    from spirob_mjlab.knockdown import playback
    clock = [0.0]
    monkeypatch.setattr(playback.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(playback.time, "sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    sim = Sim()

    class ViewerBase:
        def _execute_step(self):
            self.policy_calls += 1
            for _ in range(600):
                sim.step()
            return True

    class Viewer(PhysicsFramePlayback, ViewerBase):
        def sync_viewer_to_env(self): pass
        def sync_env_to_viewer(self): self.frames.append((sim.position, clock[0]))
        def is_running(self): return True

    viewer = Viewer()
    viewer.env = SimpleNamespace(unwrapped=SimpleNamespace(sim=sim, physics_dt=.002))
    viewer.frame_rate, viewer.frame_time = 60., 1/60
    viewer._is_paused = viewer._interrupted = False
    viewer._time_multiplier = 1.
    viewer._actions = deque()
    viewer._stats_frames = viewer.policy_calls = 0
    viewer.frames = []
    viewer._step_physics(.001)
    assert viewer.policy_calls == 1
    assert len(viewer.frames) == 72
    assert clock[0] == pytest.approx(1.2)
    assert viewer.frames[0][1] == pytest.approx(.018)
    viewer._step_physics(.001)
    assert viewer.policy_calls == 2
    assert clock[0] == pytest.approx(2.4)
    assert "step" not in vars(sim)


@pytest.mark.parametrize("camera", [False, True])
def test_standard_cli_delegates_and_scopes_viewer_override(monkeypatch, camera):
    import sys
    from mjlab.scripts import play as upstream
    from spirob_mjlab import play
    native, viser = upstream.NativeMujocoViewer, upstream.ViserPlayViewer
    task = "Mjlab-SpiRob-EggKnockdown-Camera" if camera else "Mjlab-SpiRob-EggToBucket-Stage1"
    monkeypatch.setattr(sys, "argv", ["play", task, "--agent", "random"])

    def main():
        assert issubclass(upstream.NativeMujocoViewer, PhysicsFramePlayback) == camera
        assert issubclass(upstream.ViserPlayViewer, PhysicsFramePlayback) == camera
        raise RuntimeError("finished")

    monkeypatch.setattr(upstream, "main", main)
    with pytest.raises(RuntimeError, match="finished"):
        play.main()
    assert upstream.NativeMujocoViewer is native
    assert upstream.ViserPlayViewer is viser
