"""Fast tests for the knockdown policy contract (no environment stepping).

Run: uv run pytest -q tests/test_knockdown_contract.py
"""

import ast
import inspect

import mujoco
import numpy as np
import pytest
import torch

from spirob_mjlab.knockdown import contract as C
from spirob_mjlab.knockdown import mdp
from spirob_mjlab.knockdown.entities import LAB_MODEL_XML, lab_model_sha256, lab_robot_spec
from spirob_mjlab.knockdown.env_cfg import EVAL_ABS_U, EVAL_V, TRAIN_ABS_U, TRAIN_V, knockdown_env_cfg
from spirob_mjlab.knockdown.plain_scene import build, robot_egg_contacts


# ---------------------------------------------------------------- model / frames
def test_lab_model_hash_matches_contract():
    assert lab_model_sha256() == C.LAB_MODEL_SHA256


def test_neutral_payout_and_frame_origin_match_lab_model():
    m = mujoco.MjModel.from_xml_path(str(LAB_MODEL_XML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert np.allclose(d.actuator_length, C.NEUTRAL_PAYOUT_M, atol=1e-9)
    origin = d.body("link_002").xpos
    assert np.allclose(origin[:2], C.C_ORIGIN_MODEL_XY, atol=1e-7)
    # Lab frame: u = +x, v = from base toward tip at rest = -y
    tip = d.site("tip_site").xpos
    u, v = C.model_xy_to_c(tip[0], tip[1])
    assert abs(u) < 1e-6 and v > 0.15


def test_entity_spec_equals_lab_model_except_floor():
    lab = mujoco.MjModel.from_xml_path(str(LAB_MODEL_XML))
    ent = lab_robot_spec().compile()
    assert ent.ngeom == lab.ngeom - 1  # floor plane removed only
    assert (ent.nbody, ent.njnt, ent.ntendon, ent.nu) == (lab.nbody, lab.njnt, lab.ntendon, lab.nu)
    assert np.allclose(ent.jnt_stiffness, lab.jnt_stiffness)
    assert np.allclose(ent.dof_damping, lab.dof_damping)
    assert np.isclose(ent.body_mass.sum(), lab.body_mass.sum())
    assert [ent.tendon(i).name for i in range(2)] == ["cable_0", "cable_1"]
    assert ent.nsensor == 0, "Lab model must carry no touch/force sensors"


# ---------------------------------------------------------------- actor schema
def test_actor_terms_use_only_camera_encoder_memory():
    C.validate_actor_schema()
    assert {t.source for t in C.ACTOR_TERMS} <= C.PERMITTED_ACTOR_SOURCES
    assert C.DEFAULT.actor_dim == 8


def test_actor_group_is_exactly_the_contract_term():
    cfg = knockdown_env_cfg(play=True)
    actor = cfg.observations["actor"].terms
    assert list(actor) == ["contract"]
    assert actor["contract"].func is mdp.actor_contract_obs
    assert cfg.observations["actor"].enable_corruption is False


def test_actor_function_reads_no_privileged_simulator_state():
    """Static check: the actor builder never touches sensordata, contacts, tendon_len or bodies."""
    src = inspect.getsource(mdp.actor_contract_obs) + inspect.getsource(mdp.camera_egg_uv)
    forbidden = ["sensordata", "touch", "contact", "tendon_len", "body_link", "joint_pos", "qpos", "critic"]
    names = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)} | {
        n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
    hits = [f for f in forbidden if any(f in x for x in names)]
    assert not hits, hits
    # egg_uv_true is the camera stand-in's ground truth before noise/bias; it is the only world read.
    assert "egg_uv_true" in src


# ---------------------------------------------------------------- action transform
@pytest.mark.parametrize("raw", [[0.0, 0.0], [1.0, -1.0], [-3.0, 0.5], [0.25, 2.0]])
def test_action_transform_numpy_torch_parity(raw):
    spec = C.DEFAULT.action
    prev = np.asarray(spec.neutral_m)
    for _ in range(12):
        dn = C.desired_target(np.asarray(raw), spec)
        ln = C.limited_target(dn, prev, spec)
        dt_ = C.desired_target(torch.tensor(raw, dtype=torch.float64), spec)
        lt = C.limited_target(dt_, torch.tensor(prev), spec)
        assert np.allclose(ln, lt.numpy(), atol=1e-12)
        assert np.all(np.abs(ln - prev) <= spec.max_step_m + 1e-12)
        prev = ln
    lo, hi = np.array(spec.bounds_m).T
    assert np.all(prev >= lo - 1e-12) and np.all(prev <= hi + 1e-12)


def test_action_units_order_and_sign():
    spec = C.DEFAULT.action
    d = C.desired_target(np.array([1.0, -1.0]), spec)
    assert np.isclose(d[0] - spec.neutral_m[0], +spec.half_span_m)   # cable_0 pays out
    assert np.isclose(d[1] - spec.neutral_m[1], -spec.half_span_m)   # cable_1 reels in
    assert spec.move_speed_m_s <= 0.020, "must not exceed the Lab's 20 mm/s application ceiling"
    assert spec.max_step_m / spec.move_speed_m_s < spec.decision_period_s


def test_observation_assembly_parity():
    rng = np.random.default_rng(0)
    egg, start = rng.normal(0, 0.1, 2), rng.normal(0, 0.1, 2)
    pay = np.asarray(C.NEUTRAL_PAYOUT_M) + rng.normal(0, 0.02, 2)
    prev = rng.uniform(-1, 1, 2)
    on = C.build_actor_obs(egg, start, pay, prev)
    ot = C.build_actor_obs(*(torch.tensor(x) for x in (egg, start, pay, prev)))
    assert on.shape == (8,) and np.allclose(on, ot.numpy())


# ---------------------------------------------------------------- reset validity
@pytest.mark.parametrize("region", ["train", "eval"])
def test_reset_region_has_no_initial_robot_contact(region):
    abs_u, v = (TRAIN_ABS_U, TRAIN_V) if region == "train" else (EVAL_ABS_U, EVAL_V)
    corners = [(s * a, b) for s in (-1, 1) for a in abs_u for b in v]
    rng = np.random.default_rng(1)
    random = [(s * rng.uniform(*abs_u), rng.uniform(*v)) for s in rng.choice([-1, 1], 24)]
    for uv in corners + random:
        for off in ((0.002, 0.002), (-0.002, -0.002), (0.002, -0.002)):
            m, d = build(egg_uv=uv, ref_offset_m=off)
            for _ in range(200):  # let reference offsets settle the robot (0.2 s)
                mujoco.mj_step(m, d)
            assert robot_egg_contacts(m, d) == 0, (uv, off)


def test_no_action_egg_stays_supported():
    for uv in [(0.05, 0.06), (-0.10, 0.17), (0.075, 0.115)]:
        m, d = build(egg_uv=uv)
        egg = m.body("egg/egg").id
        p0 = d.xpos[egg].copy()
        for _ in range(5000):
            mujoco.mj_step(m, d)
        assert np.linalg.norm(d.xpos[egg] - p0) < 0.001, uv
        assert np.isfinite(d.qacc).all()
