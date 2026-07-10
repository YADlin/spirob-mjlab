"""Register minimal SpiRob mjlab tasks.

The first task is deliberately small: all entities are present, the robot has
2 tendon-length actions, and PPO learns/replays a reach/contact behaviour.
"""

from mjlab.tasks.registry import register_mjlab_task

from spirob_mjlab.env_cfgs import spirob_minimal_env_cfg
from spirob_mjlab.rl_cfg import spirob_ppo_runner_cfg

TASK_ID = "Mjlab-SpiRob-Minimal"

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=spirob_minimal_env_cfg(play=False),
    play_env_cfg=spirob_minimal_env_cfg(play=True),
    rl_cfg=spirob_ppo_runner_cfg(),
)
