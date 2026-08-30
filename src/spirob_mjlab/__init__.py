"""Register the fixed-condition S1 and spawn-randomized S2 tasks."""

from mjlab.tasks.registry import register_mjlab_task

from spirob_mjlab.env_cfgs import (
    spirob_egg_to_bucket_stage1_env_cfg,
    spirob_egg_to_bucket_stage2_env_cfg,
    spirob_egg_to_bucket_stage2b_env_cfg,
    spirob_egg_to_bucket_stage2c_env_cfg,
)
from spirob_mjlab.rl_cfg import Stage2FineTuneRunner, spirob_ppo_runner_cfg

EGG_TO_BUCKET_STAGE1_TASK_ID = "Mjlab-SpiRob-EggToBucket-Stage1"
EGG_TO_BUCKET_STAGE2_TASK_ID = "Mjlab-SpiRob-EggToBucket-Stage2"
EGG_TO_BUCKET_STAGE2B_TASK_ID = "Mjlab-SpiRob-EggToBucket-Stage2B"
EGG_TO_BUCKET_STAGE2C_TASK_ID = "Mjlab-SpiRob-EggToBucket-Stage2C"

register_mjlab_task(
    task_id=EGG_TO_BUCKET_STAGE1_TASK_ID,
    env_cfg=spirob_egg_to_bucket_stage1_env_cfg(play=False),
    play_env_cfg=spirob_egg_to_bucket_stage1_env_cfg(play=True),
    rl_cfg=spirob_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id=EGG_TO_BUCKET_STAGE2_TASK_ID,
    env_cfg=spirob_egg_to_bucket_stage2_env_cfg(play=False),
    play_env_cfg=spirob_egg_to_bucket_stage2_env_cfg(play=True),
    rl_cfg=spirob_ppo_runner_cfg(stage2=True),
    runner_cls=Stage2FineTuneRunner,
)

register_mjlab_task(
    task_id=EGG_TO_BUCKET_STAGE2B_TASK_ID,
    env_cfg=spirob_egg_to_bucket_stage2b_env_cfg(play=False),
    play_env_cfg=spirob_egg_to_bucket_stage2b_env_cfg(play=True),
    rl_cfg=spirob_ppo_runner_cfg(stage2=True),
    runner_cls=Stage2FineTuneRunner,
)

register_mjlab_task(
    task_id=EGG_TO_BUCKET_STAGE2C_TASK_ID,
    env_cfg=spirob_egg_to_bucket_stage2c_env_cfg(play=False),
    play_env_cfg=spirob_egg_to_bucket_stage2c_env_cfg(play=True),
    rl_cfg=spirob_ppo_runner_cfg(stage2=True),
    runner_cls=Stage2FineTuneRunner,
)
