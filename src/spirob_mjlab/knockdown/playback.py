"""Draw real physics substeps while mjlab executes a complete policy decision.

mjlab 1.4's BaseViewer normally displays only the end of env.step(). For a
1.2-second waypoint that hides the whole movement. This mixin temporarily adds
a display callback after sim.step(), and paces those frames in wall time.
It never splits env.step(), calls the policy early, or interpolates positions.
"""
from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager

from mjlab.viewer.base import ViewerAction


@contextmanager
def physics_frames(sim, physics_dt, frame_rate, on_frame):
    """Observe integrated states without changing the simulation call sequence."""
    original = sim.step
    # Restore the instance exactly, including absence of a previous override.
    had_override = "step" in vars(sim)
    old_override = vars(sim).get("step")
    substep = 0
    frame = 1
    previous_time = 0.0

    def step():
        nonlocal substep, frame, previous_time
        original()
        substep += 1
        elapsed = substep * physics_dt
        if elapsed + 1e-10 >= frame / frame_rate:
            on_frame(elapsed, elapsed - previous_time)
            previous_time = elapsed
            frame = int((elapsed + 1e-10) * frame_rate) + 1

    sim.step = step
    try:
        yield
    finally:
        if had_override:
            sim.step = old_override
        else:
            del sim.step


class _PlaybackClosed(BaseException):
    """Leave a partially displayed decision only when closing the viewer."""


class PhysicsFramePlayback:
    """Mixin for mjlab's native and Viser viewers (play process only)."""

    def _step_physics(self, dt):
        # Frames below already pace the entire decision. The upstream accumulator
        # would add an extra 1.2-second wait before every displayed movement.
        del dt
        self._was_capped = False
        self.sync_viewer_to_env()
        self._execute_step()
        self._sim_budget = 0.0

    def _execute_step(self):
        env = self.env.unwrapped
        self._frame_deadline = time.perf_counter()
        self._displaying_single_step = self._is_paused
        try:
            with physics_frames(env.sim, env.physics_dt, self.frame_rate,
                                self._display_physics_frame):
                return super()._execute_step()
        except _PlaybackClosed:
            self._interrupted = True
            return False
        finally:
            self._last_tick_time = time.perf_counter()

    def _process_frame_controls(self):
        # Pause/speed are safe mid-decision. Reset, checkpoint switching and
        # single-step requests must wait until env.step() has completed.
        deferred = deque()
        while self._actions:
            action, payload = self._actions.popleft()
            if action == ViewerAction.TOGGLE_PAUSE:
                self.toggle_pause()
            elif action == ViewerAction.SPEED_UP:
                self.increase_speed()
            elif action == ViewerAction.SPEED_DOWN:
                self.decrease_speed()
            elif action == ViewerAction.RESET_SPEED:
                self.reset_speed()
            else:
                deferred.append((action, payload))
        self._actions.extendleft(reversed(deferred))

    def _display_physics_frame(self, elapsed, interval):
        del elapsed
        self._process_frame_controls()
        if not self._displaying_single_step:
            while self._is_paused:
                self._check_playback_open()
                self.sync_env_to_viewer()
                time.sleep(min(self.frame_time, 0.02))
                self._process_frame_controls()
                self._frame_deadline = time.perf_counter()

        # Slower machines show slower physics; never skip motion to catch up.
        self._frame_deadline = max(
            self._frame_deadline + interval / self._time_multiplier,
            time.perf_counter(),
        )
        while True:
            self._check_playback_open()
            delay = self._frame_deadline - time.perf_counter()
            if delay <= 0:
                break
            time.sleep(min(delay, 0.01))
        self.sync_env_to_viewer()
        self._stats_frames += 1

    def _check_playback_open(self):
        if self._interrupted or not self.is_running():
            raise _PlaybackClosed()
