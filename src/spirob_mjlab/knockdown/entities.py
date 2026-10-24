"""Entities for the knockdown task (assets/knockdown; the original assets are not used or changed).

Robot   : assets/knockdown/spirob_robot.xml  - NYK's 24 Oct model (source/ kept verbatim) with the
          floor plane and lights removed, formatted like assets/spirob_robot.xml (scene from mjlab).
Egg     : assets/knockdown/egg.xml       - spirob_egg.xml, black.
Pedestal: assets/knockdown/pedestal.xml  - spirob_pedestal.xml, white, 20 mm lower (fixed mocap prop).
Regenerate with tools/knockdown/make_knockdown_assets.py; MODEL_MANIFEST.json records hashes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

KD_DIR = Path(__file__).resolve().parents[1] / "assets" / "knockdown"
ROBOT_XML = KD_DIR / "spirob_robot.xml"
SOURCE_XML = KD_DIR / "source" / "spirob_robot_k2_2.148_20261024_0318.xml"
EGG_XML = KD_DIR / "egg.xml"
PEDESTAL_XML = KD_DIR / "pedestal.xml"
CABLE_NAMES = ("cable_0", "cable_1")
LINK_BODY_NAMES = tuple(f"link_{i:03d}" for i in range(1, 22))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def robot_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    if any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.worldbody.geoms):
        raise RuntimeError("Knockdown robot XML must not contain a floor; regenerate the assets")
    return spec


def robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=robot_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=CABLE_NAMES, transmission_type=TransmissionType.TENDON,
                                      command_field="position"),)),
        sort_actuators=False,
    )


def egg_cfg() -> EntityCfg:
    return EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_file(str(EGG_XML)),
                     init_state=EntityCfg.InitialStateCfg(pos=(0.08, 0.10, 0.075)))


def pedestal_cfg() -> EntityCfg:
    return EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_file(str(PEDESTAL_XML)),
                     init_state=EntityCfg.InitialStateCfg(pos=(0.08, 0.10, 0.002)))
