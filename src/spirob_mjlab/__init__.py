"""Register the fixed-condition S1 and spawn-randomized S2 tasks."""

from mjlab.tasks.registry import register_mjlab_task

from spirob_mjlab.env_cfgs import (
    spirob_egg_to_bucket_stage1_env_cfg,
    spirob_egg_to_bucket_stage2_env_cfg,
    spirob_egg_to_bucket_stage2b_env_cfg,
    spirob_egg_to_bucket_stage2c_env_cfg,
    spirob_egg_to_bucket_stage2_sector_arc1_env_cfg,
    spirob_egg_to_bucket_stage2_sector_arc2_env_cfg,
    spirob_egg_to_bucket_stage2_sector_full_env_cfg,
)
from spirob_mjlab.rl_cfg import Stage2FineTuneRunner, spirob_ppo_runner_cfg
from spirob_mjlab.sector_curriculum import (
    STAGE2_SECTOR_ARC1_TASK_ID,
    STAGE2_SECTOR_ARC2_TASK_ID,
    STAGE2_SECTOR_FULL_TASK_ID,
)

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

for task_id, env_cfg_factory in (
    (STAGE2_SECTOR_ARC1_TASK_ID, spirob_egg_to_bucket_stage2_sector_arc1_env_cfg),
    (STAGE2_SECTOR_ARC2_TASK_ID, spirob_egg_to_bucket_stage2_sector_arc2_env_cfg),
    (STAGE2_SECTOR_FULL_TASK_ID, spirob_egg_to_bucket_stage2_sector_full_env_cfg),
):
    register_mjlab_task(
        task_id=task_id,
        env_cfg=env_cfg_factory(play=False),
        play_env_cfg=env_cfg_factory(play=True),
        rl_cfg=spirob_ppo_runner_cfg(stage2=True),
        runner_cls=Stage2FineTuneRunner,
    )
