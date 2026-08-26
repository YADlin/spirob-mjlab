"""Entity definitions for SpiRob mjlab tasks.

Each detached physical object is its own EntityCfg. The robot is fixed-base and
articulated; the egg is free; the bucket is fixed. Stage 1 uses a fixed
pedestal, while Stage 2 uses a mocap variant so the pedestal and egg can share
one per-environment XY spawn offset.
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


def movable_pedestal_spec() -> mujoco.MjSpec:
    """Pedestal variant whose fixed root can be positioned per environment."""
    spec = _spec_from(SPIROB_PEDESTAL_XML)
    root_body = spec.worldbody.first_body()
    if root_body is None:
        raise ValueError("SpiRob pedestal XML has no root body")
    root_body.mocap = True
    return spec


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



def pedestal_cfg() -> EntityCfg:
    """Fixed pedestal under the default egg start pose."""
    return EntityCfg(
        spec_fn=pedestal_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.05, 0.15, 0.002)),
    )


def movable_pedestal_cfg() -> EntityCfg:
    """Mocap pedestal used only by tasks that randomize the egg spawn."""
    return EntityCfg(
        spec_fn=movable_pedestal_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.05, 0.15, 0.002)),
    )


def bucket_cfg() -> EntityCfg:
    """Fixed bucket/drop target."""
    return EntityCfg(
        spec_fn=bucket_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(-0.05, 0.15, 0.0)),
    )
