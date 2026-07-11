"""Register SpiRob mjlab tasks.

Task 1: Mjlab-SpiRob-Minimal
    Minimal reach/contact confidence-builder.

Task 2: Mjlab-SpiRob-BucketDrop
    Gravity/drop confidence-builder: all entities are present, the egg starts
    above the bucket, and the task succeeds when the egg is detected inside the
    bucket. This validates object/bucket collision, success logic, reset logic,
    vectorization, and 64-env training/play before we ask the robot to manipulate
    the egg.
"""

from mjlab.tasks.registry import register_mjlab_task

from spirob_mjlab.env_cfgs import spirob_bucket_drop_env_cfg, spirob_minimal_env_cfg
from spirob_mjlab.rl_cfg import spirob_ppo_runner_cfg

MINIMAL_TASK_ID = "Mjlab-SpiRob-Minimal"
BUCKET_DROP_TASK_ID = "Mjlab-SpiRob-BucketDrop"

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
