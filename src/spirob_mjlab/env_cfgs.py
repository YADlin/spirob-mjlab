"""mjlab environment configs for SpiRob.

This file keeps the earlier confidence tasks and adds a first deterministic
manipulation task:

    Mjlab-SpiRob-EggToBucket-Stage1

Stage 1 uses only a minimal object-level reward: reduce egg-to-bucket distance.
No randomization is used here.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.envs.mdp.actions import TendonLengthActionCfg
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
    egg_drop_cfg,
    pedestal_cfg,
    spirob_robot_cfg,
)

# Stage-1 action dynamics. Full range remains accessible; only command rate is limited.
CABLE_ACTION_SCALE_FULL_RANGE = 0.5 * (CABLE_CTRL_RANGE[1] - CABLE_CTRL_RANGE[0])
CABLE_MAX_DELTA_PER_CONTROL_STEP = 5.0e-3  # metres/control-step. 0.03 m takes ~2 s at 100 Hz.


def _robot_tip_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "robot",
        tendon_names=CABLE_NAMES,
        site_names=("tip_site",),
    )


def _bucket_site_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("bucket", site_names=("bucket_site",))


def _common_scene_entities(*, drop_task: bool = False):
    return {
        "robot": spirob_robot_cfg(),
        "pedestal": pedestal_cfg(),
        "egg": egg_drop_cfg() if drop_task else egg_cfg(),
        "bucket": bucket_cfg(),
    }


def _common_observations(*, include_touch: bool = True):
    robot_tip_cfg = _robot_tip_cfg()
    bucket_site_cfg = _bucket_site_cfg()

    actor_terms = {
        "tendon_len": ObservationTermCfg(
            func=mdp.tendon_length,
            params={"asset_cfg": robot_tip_cfg},
        ),
        "tendon_vel": ObservationTermCfg(
            func=mdp.tendon_velocity,
            params={"asset_cfg": robot_tip_cfg},
        ),
        "tip_to_egg": ObservationTermCfg(
            func=mdp.tip_to_egg,
            params={"asset_cfg": robot_tip_cfg},
        ),
        "egg_to_bucket": ObservationTermCfg(
            func=mdp.egg_to_bucket,
            params={"asset_cfg": bucket_site_cfg},
        ),
        "last_action": ObservationTermCfg(func=mdp.last_action),
    }
    if include_touch:
        actor_terms = {
            "tendon_len": actor_terms["tendon_len"],
            "tendon_vel": actor_terms["tendon_vel"],
            "touch": ObservationTermCfg(func=mdp.touch_values),
            "tip_to_egg": actor_terms["tip_to_egg"],
            "egg_to_bucket": actor_terms["egg_to_bucket"],
            "last_action": actor_terms["last_action"],
        }

    return {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=False),
        "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
    }


def _common_actions() -> dict[str, ActionTermCfg]:
    # Raw policy action is converted to desired tendon length.
    # With scale=0.045 and offset=0.22, nominal [-1,1] maps to [0.175,0.265].
    return {
        "cable_len": TendonLengthActionCfg(
            entity_name="robot",
            actuator_names=CABLE_NAMES,
            scale=0.045,
            offset=CABLE_REST,
            preserve_order=True,
            clip={name: CABLE_CTRL_RANGE for name in CABLE_NAMES},
        ),
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
        nconmax=192,
        njmax=800,
        contact_sensor_maxmatch=128,
        mujoco=MujocoCfg(
            timestep=0.0005,
            iterations=20,
            ls_iterations=20,
            impratio=10,
            cone="elliptic",
            integrator="implicitfast",
        ),
    )


def spirob_minimal_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_tip_cfg = _robot_tip_cfg()

    rewards = {
        "reach": RewardTermCfg(
            func=mdp.reach_reward,
            weight=1.0,
            params={"asset_cfg": robot_tip_cfg, "std": 0.06},
        ),
        "contact": RewardTermCfg(func=mdp.contact_reward, weight=2.0),
        "action_l2": RewardTermCfg(func=mdp.action_l2, weight=-0.005),
    }

    terminations = {
        "success_reach_or_touch": TerminationTermCfg(
            func=mdp.reached_egg,
            params={"asset_cfg": robot_tip_cfg},
        ),
        "egg_fell": TerminationTermCfg(func=mdp.egg_fell),
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
        actions=_common_actions(),
        rewards=rewards,
        terminations=terminations,
        viewer=_common_viewer_cfg(),
        sim=_common_sim_cfg(),
        decimation=20,
        episode_length_s=5.0 if not play else 1.0e9,
    )


def spirob_bucket_drop_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Passive bucket-drop confidence task."""
    bucket_site_cfg = _bucket_site_cfg()

    rewards = {
        "egg_to_bucket": RewardTermCfg(
            func=mdp.egg_to_bucket_smooth_reward,
            weight=1.0,
            params={"asset_cfg": bucket_site_cfg, "std": 0.05},
        ),
        "inside_bucket": RewardTermCfg(
            func=mdp.egg_inside_bucket_reward,
            weight=10.0,
            params={
                "asset_cfg": bucket_site_cfg,
                "com_distance_threshold": 0.015,
                "max_z_offset": 0.020,
            },
        ),
        "action_l2": RewardTermCfg(func=mdp.action_l2, weight=-0.002),
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
        "nan_state": TerminationTermCfg(func=mdp.nan_state),
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities=_common_scene_entities(drop_task=True),
            num_envs=1 if play else 64,
            env_spacing=0.70,
        ),
        observations=_common_observations(include_touch=True),
        actions=_common_actions(),
        rewards=rewards,
        terminations=terminations,
        viewer=_common_viewer_cfg(),
        sim=_common_sim_cfg(),
        decimation=20,
        episode_length_s=2.0 if not play else 1.0e9,
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

    rewards = {
        "egg_to_bucket_distance": RewardTermCfg(
            func=mdp.egg_to_bucket_distance_reward,
            weight=1.0,
            params={"asset_cfg": bucket_site_cfg, "distance_scale": 0.10},
        ),
        "inside_bucket": RewardTermCfg(
            func=mdp.egg_inside_bucket_reward,
            weight=25.0,
            params={
                "asset_cfg": bucket_site_cfg,
                "com_distance_threshold": 0.015,
                "max_z_offset": 0.020,
            },
        ),
        # Keep this tiny. Too much action penalty encourages doing nothing.
        "action_l2": RewardTermCfg(func=mdp.action_l2, weight=-0.0005),
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
        decimation=20,
        # Give the robot enough time to accidentally discover contact/motion.
        episode_length_s=8.0 if not play else 1.0e9,
    )
