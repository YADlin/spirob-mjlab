"""Check controlled configurations, XML sensor mapping, and live observation masks."""
from copy import deepcopy
from dataclasses import asdict
import argparse
import json
import types

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import spirob_mjlab
from spirob_mjlab import mdp
from spirob_mjlab.touch_ablation import TASK_IDS, ablated_touch, touch_ablation_env_cfg


def comparable(value):
    """Compare function definitions, not addresses of fresh default lambdas."""
    if hasattr(value, "__dataclass_fields__"):
        return comparable(asdict(value))
    if isinstance(value, dict):
        return {k: comparable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(comparable(v) for v in value)
    if isinstance(value, types.FunctionType):
        return (value.__module__, value.__qualname__, value.__code__,
                comparable(value.__defaults__),
                tuple(comparable(c.cell_contents) for c in (value.__closure__ or ())))
    return value


def check(device):
    a, b = [load_env_cfg(TASK_IDS[arm]) for arm in ("touch", "no-touch")]
    normalized = deepcopy(b)
    for group in ("actor", "critic"):
        normalized.observations[group].terms["touch"].params["enabled"] = True
    if comparable(a) != comparable(normalized):
        raise AssertionError("Conditions differ beyond touch visibility")
    baseline = load_env_cfg("Mjlab-SpiRob-EggToBucket-Stage1")
    for arm_cfg in (a, b):
        for field in ("scene", "events", "commands", "actions", "rewards", "terminations", "sim", "decimation"):
            if comparable(getattr(arm_cfg, field)) != comparable(getattr(baseline, field)):
                raise AssertionError(f"Baseline differs: {field}")
    for arm, cfg in zip(("touch", "no-touch"), (a, b)):
        cfg.scene.num_envs = 1
        cfg.auto_reset = False
        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        try:
            model = env.sim.mj_model
            expected = [f"touch_{i:03d}_c{c}" for i in range(1, 22) for c in (0, 1)]
            actual = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(42)]
            if [name.split("/")[-1] for name in actual] != expected:
                raise AssertionError(f"First 42 sensors are not the expected ordered touch channels: {actual}")
            np.testing.assert_array_equal(model.sensor_type[:42], int(mujoco.mjtSensor.mjSENS_TOUCH))
            np.testing.assert_array_equal(model.sensor_adr[:42], np.arange(42))
            np.testing.assert_array_equal(model.sensor_dim[:42], np.ones(42))
            for seed in (3210, 3211):
                obs, _ = env.reset(seed=seed)
                egg = env.scene["egg"]
                torch.testing.assert_close(egg.data.root_link_pos_w - env.scene.env_origins,
                                           egg.data.default_root_state[:, :3])
                for _ in range(3):
                    visible = ablated_touch(env, enabled=arm == "touch")
                    for group in ("actor", "critic"):
                        if obs[group].shape != (1, 93):
                            raise AssertionError(f"Unexpected {group} shape: {obs[group].shape}")
                        torch.testing.assert_close(obs[group][:, -42:], visible)
                    obs, *_ = env.step(torch.zeros((1, 2), device=device))
            # Nonzero synthetic readings verify masking without depending on
            # incidental contact at the nominal reset.
            from types import SimpleNamespace
            fake = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(
                sensordata=torch.tensor([[0.0, 0.0001, 0.0002] * 14], device=device))))
            assert int(ablated_touch(fake, True).sum()) == 14
            assert int(ablated_touch(fake, False).sum()) == 0
        finally:
            env.close()
    print("Touch ablation preflight: PASS (identical fixed scene; 93 actor/critic inputs; 42 touch channels)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    check(parser.parse_args().device)
