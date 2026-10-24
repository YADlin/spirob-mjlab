"""Entities for the knockdown task.

Robot: the CURRENT Lab model (spirob-cable-studio lab/model, v0.9.1), copied
byte-for-byte into ``assets/lab_model`` (see MODEL_MANIFEST.json). The only
transform applied for mjlab is removing the Lab XML's floor plane, because the
mjlab scene supplies its own terrain plane. Joints, tendons, actuators, contact
hulls and inertias are untouched; tests compare the compiled entity with the
Lab XML compiled directly.

Egg and pedestal: unchanged spirob-mjlab assets. The pedestal is fixed-base and
therefore auto-wrapped as a mocap body by mjlab, so it can be repositioned per
episode together with the egg.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from spirob_mjlab.entities import egg_spec, pedestal_spec
from spirob_mjlab.knockdown import contract as C

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
LAB_MODEL_XML = ASSET_DIR / "lab_model" / "spirob_robot.xml"
CABLE_NAMES = ("cable_0", "cable_1")
LINK_BODY_NAMES = tuple(f"link_{i:03d}" for i in range(1, 22))


def lab_model_sha256() -> str:
    return hashlib.sha256(LAB_MODEL_XML.read_bytes()).hexdigest()


def lab_robot_spec() -> mujoco.MjSpec:
    if lab_model_sha256() != C.LAB_MODEL_SHA256:
        raise RuntimeError("assets/lab_model/spirob_robot.xml differs from the contract's Lab model hash")
    spec = mujoco.MjSpec.from_file(str(LAB_MODEL_XML))
    planes = [g for g in spec.worldbody.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]
    if len(planes) != 1:
        raise RuntimeError(f"Expected exactly one floor plane in the Lab model, found {len(planes)}")
    spec.delete(planes[0])
    return spec


def lab_robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=lab_robot_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(
                    target_names_expr=CABLE_NAMES,
                    transmission_type=TransmissionType.TENDON,
                    command_field="position",
                ),
            ),
        ),
        sort_actuators=False,
    )


def egg_cfg() -> EntityCfg:
    return EntityCfg(spec_fn=egg_spec, init_state=EntityCfg.InitialStateCfg(pos=(0.08, 0.10, 0.095)))


def pedestal_cfg() -> EntityCfg:
    return EntityCfg(spec_fn=pedestal_spec, init_state=EntityCfg.InitialStateCfg(pos=(0.08, 0.10, 0.002)))
