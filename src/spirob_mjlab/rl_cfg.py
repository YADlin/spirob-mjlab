"""PPO configuration for the minimal SpiRob mjlab task."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.rl import (
    MjlabOnPolicyRunner,
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@dataclass
class Stage2FineTuneRunnerCfg(RslRlOnPolicyRunnerCfg):
    """S2 loader behavior exposed explicitly in the training CLI."""

    initialize_from_stage1: bool = True
    """Load network weights only; set False when resuming an interrupted S2 run."""


class Stage2FineTuneRunner(MjlabOnPolicyRunner):
    """Initialize S2 from R002 weights without resuming its optimizer state."""

    def __init__(self, env, train_cfg: dict, *args, **kwargs) -> None:
        train_cfg = dict(train_cfg)
        self._initialize_from_stage1 = train_cfg.pop("initialize_from_stage1")
        super().__init__(env, train_cfg, *args, **kwargs)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        fine_tune = load_cfg is None and self._initialize_from_stage1
        if fine_tune:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            }
        infos = super().load(
            path,
            load_cfg=load_cfg,
            strict=strict,
            map_location=map_location,
        )
        if fine_tune:
            self.current_learning_iteration = 0
            self.env.unwrapped.common_step_counter = 0
            print(
                "[INFO]: Initialized S2 actor/critic from checkpoint; "
                "optimizer and iteration state were reset."
            )
        return infos


def spirob_ppo_runner_cfg(
    *,
    stage2: bool = False,
) -> RslRlOnPolicyRunnerCfg:
    runner_cfg_cls = Stage2FineTuneRunnerCfg if stage2 else RslRlOnPolicyRunnerCfg
    return runner_cfg_cls(
        actor=RslRlModelCfg(
            hidden_dims=(128, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(128, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="spirob_minimal",
        save_interval=50,
        num_steps_per_env=32,
        max_iterations=1000,
    )
