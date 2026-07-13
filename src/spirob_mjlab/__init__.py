"""Register SpiRob mjlab tasks.

The package keeps earlier confidence tasks and adds the first deterministic
manipulation task.
"""

from mjlab.tasks.registry import register_mjlab_task

from spirob_mjlab.env_cfgs import (
    spirob_bucket_drop_env_cfg,
    spirob_egg_to_bucket_stage1_env_cfg,
    spirob_minimal_env_cfg,
)
from spirob_mjlab.rl_cfg import spirob_ppo_runner_cfg

MINIMAL_TASK_ID = "Mjlab-SpiRob-Minimal"
BUCKET_DROP_TASK_ID = "Mjlab-SpiRob-BucketDrop"
EGG_TO_BUCKET_STAGE1_TASK_ID = "Mjlab-SpiRob-EggToBucket-Stage1"

register_mjlab_task(
    task_id=MINIMAL_TASK_ID,
    env_cfg=spirob_minimal_env_cfg(play=False),
    play_env_cfg=spirob_minimal_env_cfg(play=True),
    rl_cfg=spirob_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id=BUCKET_DROP_TASK_ID,
    env_cfg=spirob_bucket_drop_env_cfg(play=False),
    play_env_cfg=spirob_bucket_drop_env_cfg(play=True),
    rl_cfg=spirob_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id=EGG_TO_BUCKET_STAGE1_TASK_ID,
    env_cfg=spirob_egg_to_bucket_stage1_env_cfg(play=False),
    play_env_cfg=spirob_egg_to_bucket_stage1_env_cfg(play=True),
    rl_cfg=spirob_ppo_runner_cfg(),
)
