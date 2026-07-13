"""Entity definitions for SpiRob mjlab tasks.

Each detached physical object is its own EntityCfg. The robot is fixed-base and
articulated; the egg is free; the pedestal and bucket are fixed props.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

ASSET_DIR = Path(__file__).parent / "assets"
SPIROB_ROBOT_XML = ASSET_DIR / "spirob_robot.xml"
SPIROB_EGG_XML = ASSET_DIR / "spirob_egg.xml"
SPIROB_PEDESTAL_XML = ASSET_DIR / "spirob_pedestal.xml"
SPIROB_BUCKET_XML = ASSET_DIR / "spirob_bucket.xml"

CABLE_NAMES = ("cable_0", "cable_1")
CABLE_REST = 0.22
CABLE_CTRL_RANGE = (0.15, 0.29)


def _spec_from(path: Path) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(path))


def spirob_robot_spec() -> mujoco.MjSpec:
    return _spec_from(SPIROB_ROBOT_XML)


def egg_spec() -> mujoco.MjSpec:
    return _spec_from(SPIROB_EGG_XML)


def pedestal_spec() -> mujoco.MjSpec:
    return _spec_from(SPIROB_PEDESTAL_XML)


def bucket_spec() -> mujoco.MjSpec:
    return _spec_from(SPIROB_BUCKET_XML)


def spirob_robot_cfg() -> EntityCfg:
    """Fixed-base tendon-actuated continuum robot."""
    return EntityCfg(
        spec_fn=spirob_robot_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
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
    """Free object placed on the pedestal at episode reset."""
    return EntityCfg(
        spec_fn=egg_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.05, 0.15, 0.098),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
    )


def egg_drop_cfg() -> EntityCfg:
    """Free egg initialized directly above the bucket for the bucket-drop task."""
    return EntityCfg(
        spec_fn=egg_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(-0.05, 0.15, 0.135),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
    )


def pedestal_cfg() -> EntityCfg:
    """Fixed pedestal under the default egg start pose."""
    return EntityCfg(
        spec_fn=pedestal_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.05, 0.15, 0.002)),
    )


def bucket_cfg() -> EntityCfg:
    """Fixed bucket/drop target."""
    return EntityCfg(
        spec_fn=bucket_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(-0.05, 0.15, 0.0)),
    )
