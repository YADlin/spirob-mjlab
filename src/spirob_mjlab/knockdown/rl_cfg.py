"""PPO configuration for the knockdown task (RSL-RL through mjlab).

Same algorithm settings as the Stage-1 baseline; differences are deliberate:
  - experiment_name 'spirob_knockdown' (separate log tree)
  - num_steps_per_env 24 = one full episode of decisions per rollout
  - init_std 0.6: actions are absolute targets, 1.0 saturates the clip often
  - asymmetric critic: privileged 'critic' observation group
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def knockdown_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(128, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.6, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(hidden_dims=(256, 256), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
            num_learning_epochs=5, num_mini_batches=4, learning_rate=3.0e-4, schedule="adaptive",
            gamma=0.97, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
        ),
        experiment_name="spirob_knockdown",
        save_interval=25,
        num_steps_per_env=24,
        max_iterations=300,
    )
