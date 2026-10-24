"""SpiRob egg knockdown: camera + motor-encoder policy for the Lab model.

Registered tasks
  Mjlab-SpiRob-EggKnockdown-Fixed  fixed egg placement (smoke tests, stage 0)
  Mjlab-SpiRob-EggKnockdown        randomized placement + dynamics (main task)
  Mjlab-SpiRob-EggKnockdown-Eval   held-out evaluation region (play/eval only)
"""

from mjlab.tasks.registry import register_mjlab_task

from spirob_mjlab.knockdown.env_cfg import knockdown_env_cfg
from spirob_mjlab.knockdown.rl_cfg import knockdown_ppo_runner_cfg

TASK_FIXED = "Mjlab-SpiRob-EggKnockdown-Fixed"
TASK_MAIN = "Mjlab-SpiRob-EggKnockdown"
TASK_EVAL = "Mjlab-SpiRob-EggKnockdown-Eval"

register_mjlab_task(
    task_id=TASK_FIXED,
    env_cfg=knockdown_env_cfg(fixed=True, randomize=False),
    play_env_cfg=knockdown_env_cfg(play=True, fixed=True, randomize=False),
    rl_cfg=knockdown_ppo_runner_cfg(),
)
register_mjlab_task(
    task_id=TASK_MAIN,
    env_cfg=knockdown_env_cfg(),
    play_env_cfg=knockdown_env_cfg(play=True),
    rl_cfg=knockdown_ppo_runner_cfg(),
)
register_mjlab_task(
    task_id=TASK_EVAL,
    env_cfg=knockdown_env_cfg(region="eval"),
    play_env_cfg=knockdown_env_cfg(play=True, region="eval"),
    rl_cfg=knockdown_ppo_runner_cfg(),
)

# Genuine RGB perception tasks. The original geometric-observation tasks remain
# available for fast baselines, explicitly distinguished in run/evaluation names.
for suffix, kwargs in (("", {}), ("-Fixed", {"fixed": True, "randomize": False}),
                       ("-Eval", {"region": "eval"})):
    register_mjlab_task(
        task_id="Mjlab-SpiRob-EggKnockdown-Camera" + suffix,
        env_cfg=knockdown_env_cfg(rendered_camera=True, **kwargs),
        play_env_cfg=knockdown_env_cfg(play=True, rendered_camera=True, **kwargs),
        rl_cfg=knockdown_ppo_runner_cfg(),
    )
