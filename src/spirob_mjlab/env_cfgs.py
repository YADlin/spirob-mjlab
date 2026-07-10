"""Minimal mjlab environment config for SpiRob.

This mirrors the compact Cartpole pattern: define entities, observations,
actions, rewards, terminations, then register via __init__.py. It also follows
the manipulation task pattern of keeping object entities in SceneCfg.entities.
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
    pedestal_cfg,
    spirob_robot_cfg,
)


def spirob_minimal_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
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
        "touch": ObservationTermCfg(func=mdp.touch_values),
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

    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=False),
        "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
    }

    # Raw policy action is converted to desired tendon length.
    # With scale=0.045 and offset=0.22, nominal [-1,1] maps to [0.175,0.265];
    # clip prevents Gaussian PPO exploration from exceeding XML ctrlrange.
    actions: dict[str, ActionTermCfg] = {
        "cable_len": TendonLengthActionCfg(
            entity_name="robot",
            actuator_names=CABLE_NAMES,
            scale=0.045,
            offset=CABLE_REST,
            preserve_order=True,
            clip={name: CABLE_CTRL_RANGE for name in CABLE_NAMES},
        ),
    }

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
            entities={
                "robot": spirob_robot_cfg(),
                "pedestal": pedestal_cfg(),
                "egg": egg_cfg(),
                "bucket": bucket_cfg(),
            },
            num_envs=1 if play else 64,
            env_spacing=0.70,
        ),
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="link_010",
            distance=0.75,
            elevation=-20.0,
            azimuth=120.0,
        ),
        sim=SimulationCfg(
            nconmax=128,
            njmax=700,
            contact_sensor_maxmatch=128,
            mujoco=MujocoCfg(
                timestep=0.0005,
                iterations=20,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
                integrator="implicitfast",
            ),
        ),
        decimation=20,  # 0.0005 * 20 = 100 Hz policy/control rate.
        episode_length_s=5.0 if not play else 1.0e9,
    )
