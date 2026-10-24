"""Environment configuration: Mjlab-SpiRob-EggKnockdown (contract v2).

One environment step = one policy decision (1.2 s): physics 2 ms implicitfast, decimation 600.
See docs/knockdown/DESIGN.md for the studies behind these settings.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr, time_out
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from spirob_mjlab.knockdown import contract as C
from spirob_mjlab.knockdown import mdp
from spirob_mjlab.knockdown.actions import DifferenceWaypointActionCfg
from spirob_mjlab.knockdown.entities import CABLE_NAMES, LINK_BODY_NAMES, egg_cfg, pedestal_cfg, robot_cfg

PHYSICS_DT = 2.0e-3   # knockdown timing within 0.03 s of 1 ms (docs/knockdown/DESIGN.md)
DECIMATION = int(round(C.DEFAULT.action.decision_period_s / PHYSICS_DT))

# Placement region in frame C [m] (docs/knockdown/reach_map_v2.png).
TRAIN_ABS_U = (0.045, 0.140)
TRAIN_V = (0.050, 0.180)
EVAL_ABS_U = (0.050, 0.130)
EVAL_V = (0.060, 0.170)
FIXED_UV = (0.080, 0.110)


def _keypoints() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=LINK_BODY_NAMES, preserve_order=True)


def _observations():
    return {
        "actor": ObservationGroupCfg({"contract": ObservationTermCfg(func=mdp.actor_contract_obs)}, enable_corruption=False),
        "critic": ObservationGroupCfg({"privileged": ObservationTermCfg(
            func=mdp.critic_privileged, params={"keypoint_cfg": _keypoints()})}, enable_corruption=False),
    }


def _contact_sensor() -> ContactSensorCfg:
    return ContactSensorCfg(
        name=mdp.CONTACT_SENSOR,
        primary=ContactMatch(mode="subtree", pattern="egg", entity="egg"),
        secondary=ContactMatch(mode="subtree", pattern="robot_base", entity="robot"),
        fields=("found",), reduce="none", num_slots=1, track_air_time=True)


def _events(region: str, fixed: bool, randomize: bool) -> dict[str, EventTermCfg]:
    abs_u, v = (TRAIN_ABS_U, TRAIN_V) if region == "train" else (EVAL_ABS_U, EVAL_V)
    events = {
        "reset_scene": EventTermCfg(func=reset_scene_to_default, mode="reset"),
        "episode": EventTermCfg(func=mdp.reset_knockdown_episode, mode="reset", params=dict(
            abs_u_range=abs_u, v_range=v, fixed_uv=FIXED_UV if fixed else None,
            start_shape_prob=0.7 if randomize else 0.0,
            robot_dz_range=(-0.015, 0.0) if randomize else (0.0, 0.0), yaw_range_rad=math.pi)),
    }
    if randomize:
        joints = SceneEntityCfg("robot", joint_names=(r"j_0(0[2-9]|1\d|2[01])",))
        events.update({
            "joint_stiffness": EventTermCfg(func=dr.joint_stiffness, mode="reset", params={
                "asset_cfg": joints, "ranges": (0.8, 1.25), "operation": "scale", "shared_random": True}),
            "joint_damping": EventTermCfg(func=dr.joint_damping, mode="reset", params={
                "asset_cfg": joints, "ranges": (0.5, 2.0), "operation": "scale", "shared_random": True}),
            "egg_friction": EventTermCfg(func=dr.geom_friction, mode="reset", params={
                "asset_cfg": SceneEntityCfg("egg", geom_names=(".*",)), "ranges": (0.6, 1.4),
                "operation": "scale", "axes": [0], "shared_random": True}),
            "egg_mass": EventTermCfg(func=dr.pseudo_inertia, mode="reset", params={
                "asset_cfg": SceneEntityCfg("egg", body_names=("egg",)), "alpha_range": (-0.178, 0.131)}),
        })
    return events


def knockdown_env_cfg(*, play: bool = False, region: str = "train", fixed: bool = False,
                      randomize: bool = True) -> ManagerBasedRlEnvCfg:
    spec = C.DEFAULT
    rewards = {
        "knockdown": RewardTermCfg(func=mdp.knockdown_bonus, weight=10.0),
        "approach": RewardTermCfg(func=mdp.approach_progress, weight=1.0, params={"keypoint_cfg": _keypoints()}),
        "egg_displacement": RewardTermCfg(func=mdp.egg_displacement_progress, weight=2.0),
        "command_motion": RewardTermCfg(func=mdp.command_motion_cost, weight=-0.02),
        "alive": RewardTermCfg(func=mdp.alive_cost, weight=-0.04),
    }
    terminations = {
        "numerical_fault": TerminationTermCfg(func=mdp.term_numerical_fault),
        "knocked_down": TerminationTermCfg(func=mdp.term_knocked_down),
        "spontaneous_fall": TerminationTermCfg(func=mdp.term_spontaneous_fall),
        "egg_out_of_bounds": TerminationTermCfg(func=mdp.term_egg_out_of_bounds),
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
    }
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"),
                       entities={"robot": robot_cfg(), "pedestal": pedestal_cfg(), "egg": egg_cfg()},
                       sensors=(_contact_sensor(),), num_envs=1 if play else 256, env_spacing=0.70),
        observations=_observations(),
        actions={mdp.ACTION_TERM: DifferenceWaypointActionCfg(
            entity_name="robot", actuator_names=CABLE_NAMES, preserve_order=True, spec=spec.action)},
        events=_events(region, fixed, randomize),
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="link_010",
                            distance=0.75, elevation=-60.0, azimuth=90.0),
        sim=SimulationCfg(nconmax=512, njmax=1400, contact_sensor_maxmatch=256,
                          mujoco=MujocoCfg(timestep=PHYSICS_DT, iterations=50, ls_iterations=50, impratio=10,
                                           cone="elliptic", integrator="implicitfast")),
        decimation=DECIMATION,
        episode_length_s=spec.episode_decisions * spec.action.decision_period_s,
    )
