"""Environment configuration for deterministic SpiRob egg-to-bucket manipulation.

Active task:
    Mjlab-SpiRob-EggToBucket-Stage1

The egg starts at a fixed position on the pedestal and the bucket is fixed.
No spawn or domain randomization is applied in this configuration.
"""

from __future__ import annotations
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from spirob_mjlab.actions import RateLimitedTendonLengthActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from spirob_mjlab import mdp
from spirob_mjlab.entities import (
    CABLE_CTRL_RANGE,
    CABLE_NAMES,
    CABLE_REST,
    bucket_cfg,
    egg_cfg,
    pedestal_cfg,
    spirob_robot_cfg,
)

# Stage-1 action dynamics. Full range remains accessible; only command rate is limited.
CABLE_ACTION_SCALE_FULL_RANGE = 0.5 * (CABLE_CTRL_RANGE[1] - CABLE_CTRL_RANGE[0])
CABLE_MAX_DELTA_PER_CONTROL_STEP = 1e-3
# metres/control-step; 0.03 m takes ~0.30 s at 100 Hz.

SPIROB_NUM_ELEMENTS = 21

SPIROB_ELEMENT_BODY_NAMES = tuple(
    f"link_{i:03d}"
    for i in range(1, SPIROB_NUM_ELEMENTS + 1)
)


def _robot_shape_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "robot",
        body_names=SPIROB_ELEMENT_BODY_NAMES,
        preserve_order=True,
    )

def _robot_tendon_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "robot",
        tendon_names=CABLE_NAMES,
        preserve_order=True,
    )

def _bucket_site_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("bucket", site_names=("bucket_site",))

def _common_scene_entities(*, drop_task: bool = False):
    return {
        "robot": spirob_robot_cfg(),
        "pedestal": pedestal_cfg(),
        "egg": egg_cfg(),
        "bucket": bucket_cfg(),
    }

def _common_observations(*, include_touch: bool = True,):
    robot_tendon_cfg = _robot_tendon_cfg()
    bucket_site_cfg = _bucket_site_cfg()
    robot_shape_cfg = _robot_shape_cfg()

    actor_terms = {
        "tendon_len": ObservationTermCfg(
            func=mdp.tendon_length,
            params={"asset_cfg": robot_tendon_cfg},
        ),
        "tendon_vel": ObservationTermCfg(
            func=mdp.tendon_velocity,
            params={"asset_cfg": robot_tendon_cfg},
        ),
        "egg_to_bucket": ObservationTermCfg(
            func=mdp.egg_to_bucket,
            params={"asset_cfg": bucket_site_cfg},
        ),
        "last_action": ObservationTermCfg(
            func=mdp.last_action
        ),
        "robot_keypoints_xy": ObservationTermCfg(
            func=mdp.robot_keypoints_xy,
            params={"asset_cfg": robot_shape_cfg},
        ),
    }
    if include_touch:
        actor_terms["touch"] = ObservationTermCfg(
            func=mdp.touch_values,
            )


    return {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=False),
        "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
    }

def _stage1_rate_limited_actions() -> dict[str, ActionTermCfg]:
    """Full-range tendon targets with slow command dynamics for Stage 1.

    The policy still has access to the full valid tendon-length interval
    CABLE_CTRL_RANGE = (0.15, 0.29). The command sent to the actuator can only
    change by CABLE_MAX_DELTA_PER_CONTROL_STEP at each policy/control step.
    """
    return {
        "cable_len": RateLimitedTendonLengthActionCfg(
            entity_name="robot",
            actuator_names=CABLE_NAMES,
            scale=CABLE_ACTION_SCALE_FULL_RANGE,
            offset=CABLE_REST,
            preserve_order=True,
            clip={name: CABLE_CTRL_RANGE for name in CABLE_NAMES},
            max_delta_per_step=CABLE_MAX_DELTA_PER_CONTROL_STEP,
        ),
    }

def _common_viewer_cfg() -> ViewerConfig:
    return ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="link_010",
        distance=0.75,
        elevation=-20.0,
        azimuth=120.0,
    )

def _common_sim_cfg() -> SimulationCfg:
    return SimulationCfg(
        nconmax=512,
        njmax=1400,
        contact_sensor_maxmatch=256,
        mujoco=MujocoCfg(
            timestep=0.0001,
            iterations=50,
            ls_iterations=50,
            impratio=10,
            cone="elliptic",
            integrator="implicitfast",
        ),
    )

def spirob_egg_to_bucket_stage1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Deterministic manipulation Stage 1.

    Egg starts on the pedestal. Bucket is fixed. No randomization.

    Minimal reward interpretation:
      - Reward becomes less negative as egg COM gets closer to bucket_site.
      - The robot gets no explicit reward for touching the egg yet.
      - If no learning occurs, that tells us the sparse object-motion signal is
        insufficient and we should add reach/contact shaping in Stage 2.
    """
    bucket_site_cfg = _bucket_site_cfg()
    robot_tendon_cfg = _robot_tendon_cfg()

    rewards = {
        "egg_delta_progress": RewardTermCfg(
            func=mdp.egg_to_bucket_delta_progress,
            weight=1.0,
            params={
                "asset_cfg": bucket_site_cfg,
                "progress_scale": 0.005,
            },
        ),
    
        "inside_bucket": RewardTermCfg(
            func=mdp.egg_inside_bucket_terminal_indicator,
            weight=25.0,
            params={
                "asset_cfg": bucket_site_cfg,
                "com_distance_threshold": 0.015,
                "max_z_offset": 0.02,
            },
        ),

        "egg_fell_penalty": RewardTermCfg(
            func=mdp.egg_fell_terminal_indicator,
            weight=-10.0,
        ),
    }

    terminations = {
        "success_egg_inside_bucket": TerminationTermCfg(
            func=mdp.egg_inside_bucket,
            params={
                "asset_cfg": bucket_site_cfg,
                "com_distance_threshold": 0.015,
                "max_z_offset": 0.020,
            },
        ),
        "egg_fell": TerminationTermCfg(func=mdp.egg_fell),
        "egg_oob": TerminationTermCfg(func=mdp.egg_out_of_bounds),
        "nan_state": TerminationTermCfg(func=mdp.nan_state),
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities=_common_scene_entities(drop_task=False),
            num_envs=1 if play else 64,
            env_spacing=0.70,
        ),
        observations=_common_observations(include_touch=True),
        actions=_stage1_rate_limited_actions(),
        rewards=rewards,
        terminations=terminations,
        viewer=_common_viewer_cfg(),
        sim=_common_sim_cfg(),
        decimation=100,
        # Give the robot enough time to accidentally discover contact/motion.
        episode_length_s=8.0 if not play else 1.0e9,
    )

#######################################################################