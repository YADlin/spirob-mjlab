# SpiRob mjlab — Professor Quick Start Guide

## 1. Purpose of this repository

This repository contains the ongoing implementation of the two-cable SpiRob continuum robot in **mjlab** for reinforcement-learning experiments.

The current research task is:

`Mjlab-SpiRob-EggToBucket-Stage1`

The long-term objective is for SpiRob to manipulate an egg-like object from a pedestal and place it inside a bucket.

The repository is deliberately developed in stages so that simulation validity, contact physics, observations, actions, rewards, and learned behaviour can be checked independently.

---

## 2. Current status

The following have been demonstrated successfully:

- SpiRob loads as an mjlab entity.
- Robot, pedestal, egg and bucket coexist in the vectorized mjlab scene.
- The task runs with parallel environments on GPU.
- Two tendon-length commands actuate the robot.
- The full tendon-length range remains available.
- A rate-limited command layer prevents instantaneous tendon target jumps.
- 42 touch sensors are included in the robot observations.
- Vectorized environment-coordinate issues in egg out-of-bounds checks were corrected.
- Bucket success termination and egg-fall termination operate correctly.
- PPO policies have learned purposeful motion toward the egg.
- Later experiments have learned stable contact/curling behaviour near the egg.

The complete egg-to-bucket transfer is **not yet solved repeatably**. Current work is studying how to represent useful capture/enclosure geometry before transport.

---

## 3. Software/hardware assumptions

The development environment used for the project is:

- Windows with WSL2 Ubuntu
- NVIDIA GPU accessible inside WSL
- `uv` for Python/environment management
- mjlab
- MuJoCo / MuJoCo Warp
- RSL-RL PPO
- Weights & Biases for training logs

Check the GPU inside WSL:

```bash
nvidia-smi
```

mjlab training requires a CUDA-capable NVIDIA GPU.

---

## 4. Repository layout

The custom task package is expected to look approximately like:

```text
spirob_mjlab_minimal/
├── pyproject.toml
├── README.md
├── src/
│   └── spirob_mjlab/
│       ├── __init__.py
│       ├── actions.py
│       ├── entities.py
│       ├── env_cfgs.py
│       ├── mdp.py
│       ├── rl_cfg.py
│       └── assets/
│           ├── spirob_robot.xml
│           ├── spirob_egg.xml
│           ├── spirob_pedestal.xml
│           ├── spirob_bucket.xml
│           └── meshes/
└── tools/
```

### Important files

`entities.py`
: Defines SpiRob, egg, pedestal and bucket as mjlab entities.

`env_cfgs.py`
: Defines scene, observations, actions, rewards, terminations, simulation timestep, decimation and episode length.

`actions.py`
: Contains the custom rate-limited tendon command layer.

`mdp.py`
: Contains observation, reward and termination functions.

`rl_cfg.py`
: Contains the PPO/RSL-RL configuration.

`assets/`
: Contains MuJoCo XML files and the SpiRob link meshes.

---

## 5. Clone the repositories

Clone mjlab:

```bash
git clone https://github.com/mujocolab/mjlab.git
cd mjlab
uv sync
```

Clone the SpiRob task repository either inside or beside the mjlab checkout:

```bash
git clone <SPIROB_REPOSITORY_URL>
```

Example layout:

```text
Research_Work/
├── mjlab/
└── spirob-mjlab/
```

The exact location is not important as long as the editable package is installed into the mjlab Python environment.

---

## 6. Install the SpiRob task

From the mjlab repository:

```bash
cd /path/to/mjlab

uv pip install -e /path/to/spirob-mjlab
```

Verify installation:

```bash
uv pip show spirob-mjlab
```

Verify that mjlab discovers the tasks:

```bash
uv run list-envs | grep SpiRob
```

Expected entries include tasks such as:

```text
Mjlab-SpiRob-Minimal
Mjlab-SpiRob-BucketDrop
Mjlab-SpiRob-EggToBucket-Stage1
```

---

## 7. Validate the asset files

If the repository contains the asset-validation script:

```bash
cd /path/to/spirob-mjlab
python tools/validate_assets.py
```

The script should confirm that all XML files and link meshes are available.

Do not continue if link meshes are missing.

---

## 8. First sanity test

Before RL training, run a zero-action policy:

```bash
cd /path/to/mjlab

uv run play Mjlab-SpiRob-EggToBucket-Stage1 \
  --agent zero \
  --num-envs 1
```

Then run random actions:

```bash
uv run play Mjlab-SpiRob-EggToBucket-Stage1 \
  --agent random \
  --num-envs 1
```

These tests verify that:

- the scene compiles;
- the viewer opens;
- the robot remains numerically stable;
- actions can be applied;
- the object and robot interact physically.

---

## 9. Training

A typical Stage-1 training command is:

```bash
uv run train Mjlab-SpiRob-EggToBucket-Stage1 \
  --env.scene.num-envs 256 \
  --agent.max-iterations 200
```

Longer experiments have also been run with larger iteration counts.

Training outputs are logged to **Weights & Biases (W&B)**.

The number of environments or training iterations should not be changed casually when comparing experiments. A run should have a stated hypothesis and preferably change only one main experimental variable.

---

## 10. Playing a trained policy

Locate the desired saved checkpoint, for example:

```text
model_50.pt
model_100.pt
model_150.pt
model_200.pt
```

Use the mjlab play command with the checkpoint option supported by the installed mjlab version.

To see the exact command-line option:

```bash
uv run play Mjlab-SpiRob-EggToBucket-Stage1 --help
```

Then pass the checkpoint path shown by that help output.

Intermediate checkpoints are important. The final checkpoint is not automatically the best policy.

---

## 11. Interpretation of the current RL task

The current deterministic task uses:

- fixed egg position;
- fixed pedestal;
- fixed bucket;
- no spawn randomization;
- no friction randomization.

This is intentional.

The objective at this stage is to first obtain **repeatable deterministic manipulation** before introducing domain randomization.

The research progression has been approximately:

1. Verify mjlab scene and entities.
2. Verify bucket success/reset.
3. Correct vectorized-coordinate termination errors.
4. Learn approach behaviour.
5. Penalize dropping the egg.
6. Reward safe egg-to-bucket progress.
7. Investigate capture/enclosure geometry.
8. Only after deterministic success: introduce position, friction and other randomization.

---

## 12. Action interface

SpiRob is controlled using two tendon-length targets.

The valid tendon range is approximately:

```text
0.15 m to 0.29 m
```

The full range remains accessible.

A custom action layer limits how quickly the commanded tendon length may change between control steps.

This avoids replacing a physically meaningful full-range actuator with an artificially narrow command interval.

A typical rate-limit parameter is defined in `env_cfgs.py`, for example:

```python
CABLE_MAX_DELTA_PER_CONTROL_STEP = 1e-3
```

This parameter should be interpreted together with the simulation timestep and decimation.

---

## 13. Observations

The policy currently receives information including:

- tendon lengths;
- tendon velocities;
- touch sensor states;
- vector from robot tip to egg;
- vector from egg to bucket;
- previous action;
- additional experimental capture geometry when enabled.

The robot currently has 42 touch-sensor observations distributed across the links.

---

## 14. Important experimental caution

A high reward does not automatically mean useful manipulation.

Several failure behaviours have already demonstrated this, including:

- immediate termination caused by incorrectly using world coordinates in vectorized environments;
- policies learning to drop the egg because early failure avoided future negative distance reward;
- policies learning to avoid the egg after a large fall penalty;
- policies curling near the egg and remaining there because proximity reward could be collected continuously.

Therefore every RL run should be assessed using:

1. visual play;
2. reward components;
3. termination statistics;
4. episode length;
5. physical task metrics;
6. PPO diagnostics;
7. intermediate checkpoints.

---

## 15. W&B experiment records

Training is monitored using Weights & Biases.

For important experiments, preserve:

- W&B run ID;
- Git commit;
- exact training command;
- number of environments;
- iteration count;
- random seed;
- reward configuration;
- tendon rate limit;
- evaluated checkpoint;
- observed policy behaviour.

The project includes/uses an exporter that can package W&B histories into a ZIP for independent analysis.

Do not share a W&B API key.

---

## 16. Reproducing a reported experiment

For an exact reproduction:

```bash
git checkout <COMMIT_OR_TAG>
```

Install/update the editable package:

```bash
cd /path/to/mjlab
uv pip install -e /path/to/spirob-mjlab
```

Confirm the task:

```bash
uv run list-envs | grep SpiRob
```

Then execute the exact training command recorded for that experiment.

This is preferable to attempting to reproduce a run from the latest `main` branch, because the task configuration changes as the research progresses.

---

## 17. Recommended Git workflow

`main`
: Contains reviewed research milestones.

Feature/research branches
: Contain individual hypotheses or experiments.

Examples:

```text
fix-stage1-vectorized-coordinates
stage1-terminal-outcome-rewards
stage1-safe-progress-v1
stage1-capture-geometry-v1
stage1-capture-geometry-audit
```

For a milestone:

```bash
git status
git add .
git commit -m "<clear description of milestone>"
git push
```

Then create an annotated tag:

```bash
git tag -a <TAG_NAME> -m "<description>"
git push origin <TAG_NAME>
```

Tags are useful for reproducing experiments referenced in meetings, papers or reports.

---

## 18. Current research question

The current bottleneck is no longer basic mjlab integration or approaching the object.

The research question is:

> How should useful SpiRob capture/enclosure of the egg be represented so that the learned policy progresses from one-sided contact to stable support and then directed transport toward the bucket?

The next work therefore focuses on physically meaningful capture geometry rather than simply increasing reward weights or training duration.

---

## 19. Suggested first use for a new researcher

A new user should perform only these steps initially:

```bash
# 1. Verify task registration
uv run list-envs | grep SpiRob

# 2. Open the deterministic task with no learned policy
uv run play Mjlab-SpiRob-EggToBucket-Stage1 --agent zero --num-envs 1

# 3. Run random actions
uv run play Mjlab-SpiRob-EggToBucket-Stage1 --agent random --num-envs 1
```

Only after these succeed should the user train or load a learned checkpoint.

---

## 20. Contact

For questions about:

- which Git tag corresponds to a particular result;
- W&B run IDs;
- contact-physics variants;
- interpretation of reward experiments;
- which checkpoint should be evaluated;

contact the repository maintainer/researcher before changing the environment configuration.
