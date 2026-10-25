"""Rendered sensing and regression checks (run with MUJOCO_GL=egl headlessly)."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from spirob_mjlab.knockdown import contract as C, mdp
from spirob_mjlab.knockdown import camera as camera_module
from spirob_mjlab.knockdown.camera import WIDTH, HEIGHT, ImageTracker, RenderedCameraObservation, calibration
from spirob_mjlab.knockdown.env_cfg import knockdown_env_cfg
from spirob_mjlab.knockdown.plain_scene import build


@pytest.fixture(scope="module")
def scene():
    m, d = build(camera=True)
    renderer = mujoco.Renderer(m, height=HEIGHT, width=WIDTH)
    yield m, d, renderer
    renderer.close()


@pytest.mark.parametrize("u", [-.14, -.08, -.045, .045, .08, .14])
@pytest.mark.parametrize("v", [.05, .11, .18])
def test_real_render_detection_accuracy(scene, u, v):
    m, d, renderer = scene
    q = m.jnt_qposadr[m.joint("egg/egg_free").id]
    d.qpos[q:q+3] = [*C.c_to_model_xy(u, v), mdp.EGG_SETTLED_Z]
    ped = m.body("ped/pedestal").id
    m.body_pos[ped, :2] = C.c_to_model_xy(u, v)
    mujoco.mj_forward(m, d)
    renderer.update_scene(d, camera="overhead")
    found, visible, start = ImageTracker().update(renderer.render())
    assert visible == 1
    assert np.linalg.norm(found - [u, v]) < .003, (u, v, found)
    assert np.array_equal(found, start)


def test_hidden_egg_and_reset_do_not_leak_truth(scene):
    m, d, renderer = scene
    q = m.jnt_qposadr[m.joint("egg/egg_free").id]
    d.qpos[q:q+3] = [*C.c_to_model_xy(.08, .11), mdp.EGG_SETTLED_Z]
    mujoco.mj_forward(m, d); renderer.update_scene(d, camera="overhead")
    tracker = ImageTracker()
    old, visible, start = tracker.update(renderer.render())
    assert visible == 1
    # Move egg outside the real image. Robot and its shadow must not become an egg.
    d.qpos[q:q+3] = [1, 1, .1]
    mujoco.mj_forward(m, d); renderer.update_scene(d, camera="overhead")
    last, visible, _ = tracker.update(renderer.render())
    assert visible == 0 and np.array_equal(last, old)
    tracker.reset()
    last, visible, start = tracker.update(renderer.render())
    assert visible == 0 and not last.any() and not start.any()


def test_actor_has_only_image_and_encoder_dependencies():
    # There is deliberately no egg entity, ground truth or contact data here.
    env = SimpleNamespace(num_envs=1, device="cpu", common_step_counter=0,
                          scene={"overhead_rgb": SimpleNamespace(data=SimpleNamespace(
                              rgb=torch.full((1, HEIGHT, WIDTH, 3), 255, dtype=torch.uint8)))},
                          action_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(
                              diff=torch.tensor([[.04]]), prev_action=torch.tensor([[.5]]))))
    obs = RenderedCameraObservation(None, env)(env)
    assert torch.equal(obs, torch.tensor([[0., 0., 0., 0., 0., .5, .5]]))
    cfg = knockdown_env_cfg(rendered_camera=True)
    assert cfg.observations["actor"].terms["contract"].func is RenderedCameraObservation


def test_height_specific_egg_and_cup_maps():
    cam = calibration()
    uv = np.array([.1, .17])
    pp = np.asarray(cam.principal_point_px)
    for z in (.087, .046):
        px = pp + (cam.c_to_px(uv) - pp) * (cam.camera_height_m - cam.plane_z_m)/(cam.camera_height_m-z)
        assert np.allclose(cam.px_to_c(px, feature_z_m=z), uv)
    assert cam.feature_z_m == .087


@pytest.mark.parametrize("width,height", [(1920, 1080), (1456, 819), (640, 360)])
def test_native_lab_coordinates_survive_full_image_resize(width, height):
    native = calibration(1920, 1080)
    resized = calibration(width, height)
    # 150 mm reference at the calibration plane, not the unknown screenshot height.
    points = np.array([[-.1, .05], [-.1, .2], [.14, .18]])
    source_px = native.c_to_px(points)
    assert np.linalg.norm(source_px[1] - source_px[0]) == pytest.approx(626.55)
    resized_px = (source_px + .5) * (width / 1920) - .5
    assert np.allclose(resized.c_to_px(points), resized_px, atol=1e-10)
    assert np.allclose(resized.px_to_c_plane(resized_px), points, atol=1e-12)
    # Height correction must also be independent of image resolution.
    egg_px = np.array([[380., 550.], [1450., 850.]])
    assert np.allclose(resized.px_to_c((egg_px+.5)*(width/1920)-.5), native.px_to_c(egg_px))


def test_resize_rejects_cropping_or_stretching():
    with pytest.raises(ValueError, match="field of view"):
        calibration(640, 480)
    with pytest.raises(ValueError, match="field of view"):
        calibration(0, 0)


@pytest.mark.parametrize("angle", [None, -.3, .3])
def test_mujoco_projection_matches_lab_map_including_camera_roll(monkeypatch, angle):
    # Nonzero rolls catch sign/axis errors that the almost-zero rig roll would hide.
    if angle is not None:
        monkeypatch.setitem(camera_module.PROFILE, "rotation_rad", angle)
    m, d = build(camera=True)
    cam = calibration()
    cid = m.camera("overhead").id
    rotation = d.cam_xmat[cid].reshape(3, 3)
    focal = HEIGHT / (2 * np.tan(np.radians(m.cam_fovy[cid]) / 2))
    points = np.array([[-.14, .05], [.045, .18], [.14, .11], [0., 0.]])
    for height in (.09, .087, .046):
        world = np.array([[*C.c_to_model_xy(u, v), height] for u, v in points])
        local = (world - d.cam_xpos[cid]) @ rotation
        px = np.asarray(cam.principal_point_px) + focal * local[:, :2] / -local[:, 2:3] * [1, -1]
        assert np.allclose(cam.px_to_c(px, feature_z_m=height), points, atol=1e-10)


def test_saved_lab_view_has_expected_metric_coverage():
    manifest = camera_module.camera_manifest()
    assert manifest["coverage_at_plane_m"] == pytest.approx([.45966004309312905, .2585587742398851])
    assert manifest["fovy_deg"] == pytest.approx(58.467205529828625)
    assert manifest["camera_centre_uv_m"] == pytest.approx([-.010866102256789563, .10181572478992003])


def test_floor_history_survives_separation_and_resets():
    contact = SimpleNamespace(current_contact_time=torch.tensor([[0.], [.002], [0.]]),
                              last_contact_time=torch.tensor([[.002], [0.], [0.]]))
    env = SimpleNamespace(scene={mdp.FLOOR_CONTACT_SENSOR: SimpleNamespace(data=contact)})
    assert mdp.egg_on_floor(env).tolist() == [True, True, False]
    contact.last_contact_time.zero_(); contact.current_contact_time.zero_()
    assert not mdp.egg_on_floor(env).any()


def test_terminal_precedence(monkeypatch):
    monkeypatch.setattr(mdp, "numerical_fault", lambda _: torch.tensor([True, False, False, False]))
    monkeypatch.setattr(mdp, "egg_on_floor", lambda _: torch.tensor([True, True, True, False]))
    monkeypatch.setattr(mdp, "egg_touched", lambda _: torch.tensor([True, True, False, False]))
    monkeypatch.setattr(mdp, "egg_pos_w", lambda _: torch.ones(4, 3))
    env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(4, 3)),
                          episode_length_buf=torch.full((4,), 30), max_episode_length=30)
    assert mdp.term_numerical_fault(env).tolist() == [True, False, False, False]
    assert mdp.term_knocked_down(env).tolist() == [False, True, False, False]
    assert mdp.term_spontaneous_fall(env).tolist() == [False, False, True, False]
    assert mdp.term_egg_out_of_bounds(env).tolist() == [False, False, False, True]
    assert not mdp.term_time_out(env).any()
