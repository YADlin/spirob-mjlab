"""Real RGB camera observations: MuJoCo image -> Lab detector -> seven-input actor.

No segmentation, depth or object coordinates enter the detector. Geometry is
used only to configure the fixed camera and its calibration. MuJoCo-Warp renders
the training sensor; MuJoCo's OpenGL renderer is used by the CPU demo/tests.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
import torch

from mjlab.entity import EntityCfg
from mjlab.sensor import CameraSensorCfg
from spirob_mjlab.knockdown import contract as C
from spirob_mjlab.knockdown.camera_geometry import CameraMap
from spirob_mjlab.knockdown.egg_vision import Colour, EggDetector

SENSOR_NAME = "overhead_rgb"
CAMERA_NAME = "overhead"
WIDTH, HEIGHT = 640, 360
PLANE_Z = 0.09
EGG_FEATURE_Z = 0.087
FOVY = math.degrees(2 * math.atan(C.DEFAULT.camera.visible_half_extent_uv_m[1]
                                / (C.DEFAULT.camera.height_m - PLANE_Z)))
FLOOR_RGBA = (0.50, 0.38, 0.26, 1.0)


def calibration(width=WIDTH, height=HEIGHT, fovy=FOVY) -> CameraMap:
    focal_px = height / (2 * math.tan(math.radians(fovy) / 2))
    scale = focal_px / (C.DEFAULT.camera.height_m - PLANE_Z)
    # MuJoCo image coordinates refer to pixel centres, not their edges.
    pp = np.array([(width - 1) / 2, (height - 1) / 2])
    translation = pp - scale * np.asarray(C.DEFAULT.camera.centre_uv_m)
    return CameraMap(scale, 0.0, tuple(translation), PLANE_Z,
                     C.DEFAULT.camera.height_m, tuple(pp), EGG_FEATURE_Z)


def detector(camera: CameraMap) -> EggDetector:
    return EggDetector(Colour(0, 179, 0, 255, 0, 80),
                       camera.pixels_per_m_at(camera.feature_z_m))


def add_camera(spec: mujoco.MjSpec, parent=None):
    parent = spec.worldbody if parent is None else parent
    u, v = C.DEFAULT.camera.centre_uv_m
    x, y = C.c_to_model_xy(u, v)
    return parent.add_camera(name=CAMERA_NAME, pos=(x, y, C.DEFAULT.camera.height_m),
                             quat=(1, 0, 0, 0), fovy=FOVY, resolution=(WIDTH, HEIGHT))


def camera_entity() -> EntityCfg:
    def spec_fn():
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="camera_rig", mocap=True)
        add_camera(spec, body)
        return spec
    return EntityCfg(spec_fn=spec_fn)


def camera_sensor() -> CameraSensorCfg:
    return CameraSensorCfg(name=SENSOR_NAME, camera_name="camera/" + CAMERA_NAME,
                           width=WIDTH, height=HEIGHT, data_types=("rgb",),
                           use_textures=False, use_shadows=True)


def style_scene(spec: mujoco.MjSpec):
    spec.geom("terrain").rgba = FLOOR_RGBA
    for geom in spec.geoms:
        if geom.name.startswith("robot/"):
            geom.rgba = (0.9, 0.9, 0.9, 1.0)


class ImageTracker:
    """Image-only state, also usable by CPU replay. Missing detections never mean success."""
    def __init__(self, camera=None):
        self.camera = camera or calibration()
        self.detector = detector(self.camera)
        self.reset()

    def reset(self):
        self.last_seen = np.zeros(2)
        self.start = np.zeros(2)
        self.started = False
        self.visible = False
        self.detection = None

    def update(self, rgb):
        det = self.detector.detect(rgb)
        self.detection = det
        self.visible = det is not None
        if det is not None:
            uv = self.camera.px_to_c(det.px)
            if not np.isfinite(uv).all():
                raise ValueError("Non-finite camera detection")
            self.last_seen = uv
            if not self.started:
                self.start = uv.copy()
                self.started = True
        return self.last_seen.copy(), float(self.visible), self.start.copy()


class RenderedCameraObservation:
    """mjlab observation term. Camera tensors are the only object input."""
    def __init__(self, cfg, env):
        self.trackers = [ImageTracker() for _ in range(env.num_envs)]
        self.device = env.device
        self.cached = None
        self.stamp = None

    def reset(self, env_ids=None):
        ids = range(len(self.trackers)) if env_ids is None else env_ids.tolist()
        for i in ids:
            self.trackers[i].reset()
        self.stamp = None

    def __call__(self, env):
        # Multiple observation reads at the same decision consume the same frame.
        stamp = env.common_step_counter
        if stamp != self.stamp or self.cached is None:
            frames = env.scene[SENSOR_NAME].data.rgb
            if frames is None:
                raise RuntimeError("RGB camera is unavailable; no truth-coordinate fallback")
            rows = [t.update(rgb) for t, rgb in zip(self.trackers, frames.detach().cpu().numpy(), strict=True)]
            self.cached = (
                torch.tensor(np.array([r[0] for r in rows]), device=self.device, dtype=torch.float32),
                torch.tensor([[r[1]] for r in rows], device=self.device, dtype=torch.float32),
                torch.tensor(np.array([r[2] for r in rows]), device=self.device, dtype=torch.float32),
            )
            self.stamp = stamp
        act = env.action_manager.get_term("cable_diff")
        return C.build_actor_obs(*self.cached, act.diff, act.prev_action)
