"""mjlab environment configs for SpiRob.

The file intentionally keeps two tasks side-by-side:
  * spirob_minimal_env_cfg      : first working reach/contact task
  * spirob_bucket_drop_env_cfg  : next confidence-builder for bucket success

Both use the same robot action interface so later tasks can increase difficulty
without changing the policy/action wiring.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.envs.mdp.actions import TendonLengthActionCfg
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


def _common_scene_entities(*, drop_task: bool = False):
    return {
        "robot": spirob_robot_cfg(),
        "pedestal": pedestal_cfg(),
        "egg": egg_drop_cfg() if drop_task else egg_cfg(),
        "bucket": bucket_cfg(),
    }


def _common_actions() -> dict[str, ActionTermCfg]:
    # Raw policy action is converted to desired tendon length.
    # With scale=0.045 and offset=0.22, nominal [-1,1] maps to [0.175,0.265].
    # clip prevents Gaussian PPO exploration from exceeding XML ctrlrange.
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


def _common_viewer_cfg() -> ViewerConfig:
    return ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="link_010",
        distance=0.75,
        elevation=-20.0,
        azimuth=120.0,
    )


def _common_observations(*, include_touch: bool = True):
    robot_tip_cfg = SceneEntityCfg(
        "robot",
        tendon_names=CABLE_NAMES,
        site_names=("tip_site",),
    )
    bucket_site_cfg = SceneEntityCfg("bucket", site_names=("bucket_site",))

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
        # Put touch after tendon state for compatibility with the original minimal task.
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


def spirob_minimal_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_tip_cfg = SceneEntityCfg(
        "robot",
        tendon_names=CABLE_NAMES,
        site_names=("tip_site",),
    )

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
        decimation=20,  # 0.0005 * 20 = 100 Hz policy/control rate.
        episode_length_s=5.0 if not play else 1.0e9,
    )


def spirob_bucket_drop_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Bucket-drop confidence task.

    The robot, pedestal, bucket, and egg are all present. The egg starts above the
    bucket, so zero actions should be able to produce success if the bucket and
    egg collide properly. This validates the scene, collision masks, vectorized
    reset, success detection, and PPO runner before we add manipulation difficulty.
    """
    bucket_site_cfg = SceneEntityCfg("bucket", site_names=("bucket_site",))

    rewards = {
        # Dense shaping: egg closer to bucket center is better.
        "egg_to_bucket": RewardTermCfg(
            func=mdp.egg_to_bucket_reward,
            weight=1.0,
            params={"asset_cfg": bucket_site_cfg, "std": 0.05},
        ),
        # Large sparse success bonus.
        "inside_bucket": RewardTermCfg(
            func=mdp.egg_inside_bucket_reward,
            weight=10.0,
            params={
                "asset_cfg": bucket_site_cfg,
                "com_distance_threshold": 0.015,
                "max_z_offset": 0.020,
            },
        ),
        # Keep policy quiet; the robot is present but this stage is about scene mechanics.
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
        "egg_missed_bucket": TerminationTermCfg(
            func=mdp.egg_missed_bucket,
            params={"asset_cfg": bucket_site_cfg},
        ),
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
        # A short episode is enough for gravity drop; keep play unlimited for visual inspection.
        episode_length_s=2.0 if not play else 1.0e9,
    )
