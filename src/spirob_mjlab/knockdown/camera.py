"""Real RGB camera observations: MuJoCo image -> Lab detector -> seven-input actor.

No segmentation, depth or object coordinates enter the detector. Geometry is
used only to configure the fixed camera and its calibration. MuJoCo-Warp renders
the training sensor; MuJoCo's OpenGL renderer is used by the CPU demo/tests.
"""
from __future__ import annotations

import math
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

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
PROFILE_PATH = Path(__file__).with_name("camera_profile.json")
PROFILE = json.loads(PROFILE_PATH.read_text())
NATIVE_WIDTH, NATIVE_HEIGHT = PROFILE["native_resolution_px"]
WIDTH, HEIGHT = PROFILE["render_resolution_px"]
PLANE_Z = PROFILE["plane_z_m"]
EGG_FEATURE_Z = PROFILE["feature_z_m"]
FOCAL_NATIVE_PX = PROFILE["pixels_per_m"] * (PROFILE["camera_height_m"] - PLANE_Z)
FOVY = math.degrees(2 * math.atan(NATIVE_HEIGHT / (2 * FOCAL_NATIVE_PX)))
FLOOR_RGBA = (0.50, 0.38, 0.26, 1.0)


def calibration(width=WIDTH, height=HEIGHT) -> CameraMap:
    """Resize the saved native alignment without changing its physical view.

    Native Lab coordinates are pixel centres. For a full-image resize by r,
    p_new = (p_old + 0.5) * r - 0.5, for both translation and principal point.
    Cropping or changing aspect ratio needs a new calibration, not stretching.
    """
    ratio = width / NATIVE_WIDTH
    if width <= 0 or height <= 0 or not math.isclose(ratio, height / NATIVE_HEIGHT):
        raise ValueError("Camera resolution must preserve the native 16:9 field of view")
    translation = (np.asarray(PROFILE["translation_px"]) + 0.5) * ratio - 0.5
    pp = ((width - 1) / 2, (height - 1) / 2)
    return CameraMap(PROFILE["pixels_per_m"] * ratio, PROFILE["rotation_rad"],
                     tuple(translation), PLANE_Z, PROFILE["camera_height_m"], pp, EGG_FEATURE_Z)


def camera_manifest() -> dict:
    """Bind an evaluation/export to the actual rendered view, not only the detector."""
    cam = calibration()
    native = calibration(NATIVE_WIDTH, NATIVE_HEIGHT)
    value = dict(profile=PROFILE, render_map=asdict(cam),
                 camera_centre_uv_m=cam.px_to_c_plane(cam.principal_point_px).tolist(),
                 fovy_deg=FOVY,
                 fovx_deg=math.degrees(2 * math.atan(NATIVE_WIDTH / (2 * FOCAL_NATIVE_PX))),
                 coverage_at_plane_m=[NATIVE_WIDTH / native.pixels_per_m,
                                      NATIVE_HEIGHT / native.pixels_per_m],
                 source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), PROFILE_PATH)})
    return json.loads(json.dumps(value))  # JSON-native lists for exact manifest comparisons


def detector(camera: CameraMap) -> EggDetector:
    return EggDetector(Colour(0, 179, 0, 255, 0, 80),
                       camera.pixels_per_m_at(camera.feature_z_m))


def add_camera(spec: mujoco.MjSpec, parent=None):
    parent = spec.worldbody if parent is None else parent
    cam = calibration()
    u, v = cam.px_to_c_plane(cam.principal_point_px)
    x, y = C.c_to_model_xy(u, v)
    # Frame C has v = model -y, so a camera world-Z roll of theta produces
    # the same image-plane R(theta) used by the Lab's similarity calibration.
    half_angle = cam.rotation_rad / 2
    return parent.add_camera(name=CAMERA_NAME, pos=(x, y, cam.camera_height_m),
                             quat=(math.cos(half_angle), 0, 0, math.sin(half_angle)),
                             fovy=FOVY, resolution=(WIDTH, HEIGHT))


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
        print(f"[SpiRob camera] Lab {NATIVE_WIDTH}x{NATIVE_HEIGHT} calibration -> "
              f"{WIDTH}x{HEIGHT} RGB; vertical FOV {FOVY:.3f} deg; "
              "full-image resize, same physical field of view", flush=True)

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
