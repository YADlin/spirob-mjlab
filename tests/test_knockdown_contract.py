"""Fast tests for the knockdown contract v2 and assets (no environment stepping).

Run: uv run --with pytest pytest -q tests/test_knockdown_contract.py
"""

import ast
import inspect
import json

import mujoco
import numpy as np
import pytest
import torch

from spirob_mjlab.knockdown import contract as C
from spirob_mjlab.knockdown import mdp
from spirob_mjlab.knockdown.entities import EGG_XML, PEDESTAL_XML, ROBOT_XML, SOURCE_XML, robot_spec, sha256
from spirob_mjlab.knockdown.env_cfg import EVAL_ABS_U, EVAL_V, TRAIN_ABS_U, TRAIN_V, knockdown_env_cfg
from spirob_mjlab.knockdown.plain_scene import build, robot_egg_contacts

ASSETS = ROBOT_XML.parents[1]


# ---------------------------------------------------------------- assets
def test_source_model_hash_and_originals_untouched():
    assert sha256(SOURCE_XML).startswith(C.SOURCE_MODEL_SHA256)
    # originals stay byte-identical to the upstream assets
    assert sha256(ASSETS / "spirob_robot.xml").startswith("f62de0e026cc")
    assert sha256(ASSETS / "spirob_egg.xml") == sha256(EGG_XML.parent.parent / "spirob_egg.xml")


def test_knockdown_robot_equals_source_minus_scene_items():
    s = mujoco.MjSpec.from_file(str(SOURCE_XML))
    s.meshdir = str(ROBOT_XML.parent)          # source/ has no meshes folder; resolve against knockdown/
    src = s.compile()
    kd = robot_spec().compile()
    assert kd.ngeom == src.ngeom - 1 and kd.nlight == 0 and src.nlight == 3
    assert (kd.nbody, kd.njnt, kd.ntendon, kd.nu, kd.nsensor) == (src.nbody, src.njnt, src.ntendon, src.nu, 0)
    assert np.allclose(kd.jnt_stiffness, src.jnt_stiffness) and np.allclose(kd.dof_damping, src.dof_damping)
    assert np.allclose(kd.body_mass, src.body_mass) and np.allclose(kd.body_pos, src.body_pos)
    assert np.isclose(kd.jnt_stiffness[1], 2.1480704)


def test_old_and_knockdown_robot_share_names():
    old = mujoco.MjModel.from_xml_path(str(ASSETS / "spirob_robot.xml"))
    kd = robot_spec().compile()
    names = lambda m, t, n: [mujoco.mj_id2name(m, t, i) for i in range(n)]  # noqa: E731
    for t, attr in ((mujoco.mjtObj.mjOBJ_BODY, "nbody"), (mujoco.mjtObj.mjOBJ_JOINT, "njnt"),
                    (mujoco.mjtObj.mjOBJ_TENDON, "ntendon"), (mujoco.mjtObj.mjOBJ_ACTUATOR, "nu")):
        assert names(old, t, getattr(old, attr)) == names(kd, t, getattr(kd, attr))


def test_neutral_mean_and_frame_origin():
    m = robot_spec().compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert np.allclose(d.ten_length.mean(), C.MEAN_PAYOUT_M, atol=1e-12)
    assert np.allclose(d.body("link_002").xpos[:2], C.C_ORIGIN_MODEL_XY, atol=1e-7)


def test_pedestal_lowered_and_coloured():
    s = mujoco.MjSpec.from_file(str(PEDESTAL_XML)).compile()
    d = mujoco.MjData(s)
    mujoco.mj_forward(s, d)
    cup = s.geom("ped_cup").id
    top = d.geom_xpos[cup][2] + s.geom_size[cup][2] + mdp.PEDESTAL_ROOT_Z
    assert np.isclose(top, C.DEFAULT.success.sim_support_top_z_m, atol=1e-9)
    assert np.allclose(s.geom_rgba[cup][:3], [0.93, 0.93, 0.93])
    egg = mujoco.MjSpec.from_file(str(EGG_XML)).compile()
    assert np.allclose(egg.geom_rgba[:, :3], 0.06)


def test_pose_table_matches_model():
    t = json.loads(mdp.POSE_TABLE.read_text())
    assert t["model_sha256"] == sha256(ROBOT_XML)
    diffs = [r["diff_mm"] for r in t["rows"]]
    assert diffs[0] == -80 and diffs[-1] == 80 and 0 in diffs
    straight = t["rows"][diffs.index(0)]
    assert abs(straight["tip_uv"][0]) < 1e-6 and straight["tip_uv"][1] > 0.2


# ---------------------------------------------------------------- actor schema
def test_actor_terms_camera_encoder_memory_only():
    C.validate_actor_schema()
    assert C.DEFAULT.actor_dim == 7


def test_actor_group_is_exactly_the_contract_term():
    cfg = knockdown_env_cfg(play=True)
    assert list(cfg.observations["actor"].terms) == ["contract"]
    assert cfg.observations["actor"].terms["contract"].func is mdp.actor_contract_obs


def test_actor_function_reads_no_privileged_state():
    src = inspect.getsource(mdp.actor_contract_obs) + inspect.getsource(mdp.camera_egg)
    names = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)} | {
        n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
    for bad in ("sensordata", "touch", "contact", "tendon_len", "body_link", "joint_pos", "qpos", "critic"):
        assert not any(bad in x for x in names), bad


# ---------------------------------------------------------------- action transform
@pytest.mark.parametrize("raw", [0.0, 1.0, -3.0, 0.004, 0.37])
def test_action_numpy_torch_parity_and_limits(raw):
    a = C.DEFAULT.action
    prev_n = np.array([0.0]); prev_t = torch.tensor([0.0], dtype=torch.float64)
    for _ in range(25):
        ln = C.limited_diff(C.desired_diff(np.array([raw]), a), prev_n, a)
        lt = C.limited_diff(C.desired_diff(torch.tensor([raw], dtype=torch.float64), a), prev_t, a)
        assert np.allclose(ln, lt.numpy())
        assert abs(ln[0] - prev_n[0]) <= a.max_diff_step_m + 1e-12 and abs(ln[0]) <= a.diff_limit_m + 1e-12
        prev_n, prev_t = ln, lt
    if raw == 0.004:   # 0.32 mm request is inside the deadband: no move
        assert prev_n[0] == 0.0


def test_payouts_and_speed_rules():
    a = C.DEFAULT.action
    p = C.payouts_from_diff(np.array([0.04]))
    assert np.allclose(p, [a.mean_m + 0.02, a.mean_m - 0.02]) and np.isclose(C.diff_from_payouts(p)[0], 0.04)
    assert np.allclose(C.payouts_from_diff(torch.tensor([[0.04]])).numpy(), [[a.mean_m + 0.02, a.mean_m - 0.02]])
    # speed is proportional to the step: a full 8 mm step runs at the ceiling, 1 mm at 1/8 of it
    for v in (0.005, 0.010, 0.015):
        assert np.isclose(C.cable_speed(np.array(0.008), v), v)
        assert np.isclose(C.cable_speed(np.array(0.001), v), max(v / 8, a.min_cable_speed_m_s))
    assert np.isclose(C.cable_speed(np.array(0.001), 0.015), 0.001875)
    assert a.train_cable_speed_range_m_s[1] <= 0.015 + 1e-12


def test_observation_parity():
    rng = np.random.default_rng(0)
    args = (rng.normal(0, .1, 2), np.array([1.0]), rng.normal(0, .1, 2), np.array([0.03]), np.array([0.2]))
    on = C.build_actor_obs(*args)
    ot = C.build_actor_obs(*(torch.tensor(x) for x in args))
    assert on.shape == (7,) and np.allclose(on, ot.numpy())


# ---------------------------------------------------------------- physics checks (CPU MuJoCo)
def _table():
    return json.loads(mdp.POSE_TABLE.read_text())


@pytest.mark.parametrize("region", ["train", "eval"])
def test_placements_clear_of_robot_in_start_shape(region):
    abs_u, v = (TRAIN_ABS_U, TRAIN_V) if region == "train" else (EVAL_ABS_U, EVAL_V)
    t = _table()
    rng = np.random.default_rng(3)
    checked = 0
    for row in t["rows"][::8]:
        for _ in range(6):
            uv = (rng.choice([-1, 1]) * rng.uniform(*abs_u), rng.uniform(*v))
            if C.robot_clearance(uv, row["diff_mm"] / 1000, t) <= C.DEFAULT.autostart.clearance_m:
                continue
            m, d = build(egg_uv=uv, diff_m=row["diff_mm"] / 1000, qpos=row["qpos"])
            for _ in range(200):
                mujoco.mj_step(m, d)
            assert robot_egg_contacts(m, d) == 0, (uv, row["diff_mm"])
            checked += 1
    assert checked >= 20


def test_no_action_egg_stays_supported_from_curled_starts():
    t = _table()
    for diff_mm, uv in ((0, (0.06, 0.06)), (40, (0.10, 0.17)), (-60, (-0.05, 0.15))):
        row = next(r for r in t["rows"] if r["diff_mm"] == diff_mm)
        if C.robot_clearance(uv, diff_mm / 1000, t) <= C.DEFAULT.autostart.clearance_m:
            continue
        m, d = build(egg_uv=uv, diff_m=diff_mm / 1000, qpos=row["qpos"])
        egg = m.body("egg/egg").id
        p0 = d.xpos[egg].copy()
        for _ in range(4000):
            mujoco.mj_step(m, d)
        assert np.linalg.norm(d.xpos[egg] - p0) < 0.001 and np.isfinite(d.qacc).all(), diff_mm
