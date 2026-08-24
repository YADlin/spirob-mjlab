# SpiRob–MJLab

GPU-accelerated reinforcement-learning environment for a fixed-base,
two-cable SpiRob that must move an egg from a pedestal into a bucket.

The project is implemented as its own Python package. `mjlab` is a pinned
dependency of this package, so users work inside this repository; they do not
need to clone or modify the `mjlab` source tree.

## Current research status

The registered task is:

```text
Mjlab-SpiRob-EggToBucket-Stage1
```

Stage 1 uses a fixed robot base, fixed egg start position, and fixed bucket
position. There is no spawn randomization or dynamics randomization in the
current task.

Run R002 demonstrated that the present observation/action design and a simple
object-level reward can produce egg-to-bucket placement in this fixed scene.
This is evidence that the task can be learned; it is not yet evidence of
multi-seed reliability, robustness to changed object positions, or sim-to-real
readiness. See [the R002 run record](docs/runs/R002.md) for the exact claim and
its limitations.

The repository contains the environment, assets, PPO configuration, locked
dependencies, and run documentation. It does **not** currently contain a
pretrained policy checkpoint.

## What the task contains

| Part | Current Stage-1 definition |
| --- | --- |
| Robot | Fixed-base SpiRob with 21 compliant elements |
| Action | Two tendon-length targets: `cable_0` and `cable_1` |
| Tendon range | 0.15–0.29 m; rest command 0.22 m |
| Command limit | Maximum change of 0.001 m per 0.01 s control step |
| Actor observation | 93 values: tendon state, egg-to-bucket vector, previous action, 21 XY robot keypoints, and 42 binary touch values |
| Reward | Signed egg progress toward the bucket, +25 terminal success, and −10 terminal egg fall |
| Terminations | Egg inside bucket, egg fall, egg out of bounds, non-finite state, or 8 s timeout |
| Physics/control | 0.0001 s physics step, 100 physics steps per action, 100 Hz control |
| Training | PPO through RSL-RL; 64 parallel environments by default |

## Recommended system

For training, use:

- Ubuntu Linux, either native or through WSL2 on Windows;
- an NVIDIA GPU visible from Linux/WSL;
- a recent NVIDIA driver compatible with the CUDA 13 runtime packages locked
  by this repository;
- Git and `uv`;
- several gigabytes of free disk space for PyTorch and CUDA dependencies.

The accepted R002 setup used Ubuntu under WSL2, Python 3.12.3, mjlab 1.4.0,
MuJoCo 3.8.1, MuJoCo Warp 3.8.1, Warp 1.14.0, and an NVIDIA RTX 3070 Ti
Laptop GPU.

CPU execution can be useful for limited inspection, but GPU training is the
supported workflow for this project. Direct native-Windows training has not
been validated here.

## Quick start

### 1. Check the GPU

Run this in the Linux or WSL terminal:

```bash
nvidia-smi
```

Do not continue to training until this command detects the NVIDIA GPU. On a
Windows machine, install WSL2, an Ubuntu distribution, and an NVIDIA driver
with WSL support first.

### 2. Install Git, curl, and uv

On Ubuntu or WSL:

```bash
sudo apt update
sudo apt install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen the terminal after installing `uv`, then verify it:

```bash
uv --version
```

### 3. Clone this repository

For WSL, keep the repository in the Linux filesystem—for example under
`~/Research_Work`—rather than under `/mnt/c` or `/mnt/d`.

```bash
mkdir -p ~/Research_Work
cd ~/Research_Work
git clone https://github.com/YADlin/spirob-mjlab.git
cd spirob-mjlab
git switch main
```

Record the version you are about to run:

```bash
git status --short --branch
git rev-parse HEAD
git describe --tags --always
```

### 4. Create the locked environment

The repository requests Python 3.12.3 through `.python-version`. Let `uv`
install that interpreter and synchronize the exact packages in `uv.lock`:

```bash
uv python install 3.12.3
uv sync --frozen
```

Use `uv run ...` for every project command below. There is no need to activate
`.venv` manually.

### 5. Verify Python, CUDA, assets, and task registration

```bash
uv run python --version
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
uv run python -c "import warp as wp; wp.init(); print('Warp devices:', wp.get_devices())"
uv run python tools/validate_assets.py
uv run list-envs | grep SpiRob
```

Expected final output from the task check:

```text
Mjlab-SpiRob-EggToBucket-Stage1
```

The asset validator should report `PASS`. All 21 STL files are already
included in the repository; a new user should not need to copy meshes by hand.

## Run the simulator before training

Always test the environment with non-learning agents first. These checks
separate installation or physics failures from PPO behavior.

Zero-action check:

```bash
uv run play Mjlab-SpiRob-EggToBucket-Stage1 \
  --agent zero \
  --num-envs 1 \
  --viewer auto
```

Random-action check:

```bash
uv run play Mjlab-SpiRob-EggToBucket-Stage1 \
  --agent random \
  --num-envs 1 \
  --viewer auto
```

Confirm that the scene loads, the robot and both objects are visible, tendon
commands change smoothly, and the simulation does not immediately produce a
NaN or repeated reset.

`--viewer auto` uses the native MuJoCo window when a display is available. If
no window appears in WSL, on a headless workstation, or over SSH, use the
browser viewer:

```bash
uv run play Mjlab-SpiRob-EggToBucket-Stage1 \
  --agent zero \
  --num-envs 1 \
  --viewer viser
```

Viser normally serves the viewer at `http://localhost:8080`.

## First training run: short smoke test

This 20-iteration run is only a software and stability check. It is not long
enough to judge whether the policy learns the task.

```bash
uv run train Mjlab-SpiRob-EggToBucket-Stage1 \
  --env.scene.num-envs 64 \
  --agent.max-iterations 20 \
  --agent.seed 42 \
  --agent.logger tensorboard \
  --agent.run-name smoke-seed42
```

The smoke test passes if:

- environment creation succeeds on the GPU;
- the run completes without an exception;
- no mass out-of-bounds termination occurs at reset;
- rewards, episode lengths, losses, and termination terms are logged;
- no NaN propagates into the policy.

Local outputs are written under:

```text
logs/rsl_rl/spirob_minimal/TIMESTAMP_smoke-seed42/
```

`spirob_minimal` is the current legacy experiment-directory name; it does not
mean that the old minimal reach/contact task is still active.

To inspect local TensorBoard logs:

```bash
uv run tensorboard --logdir logs/rsl_rl
```

Then open `http://localhost:6006`.

## Reproduce the R002 training configuration

R002 used 256 environments, 500 PPO iterations, and seed 42. Its accepted
source is tagged `r002-stage1-fixed-scene-v1`.

For an exact source checkout:

```bash
git switch --detach r002-stage1-fixed-scene-v1
uv sync --frozen
uv run python tools/validate_assets.py
```

The current `main` branch points to this tagged baseline at the time of this
README update. The detached checkout above prevents later changes to `main`
from silently changing a reproduction.

### Option A: Weights & Biases logging

The PPO configuration uses Weights & Biases by default.

```bash
uv run wandb login

uv run train Mjlab-SpiRob-EggToBucket-Stage1 \
  --env.scene.num-envs 256 \
  --agent.max-iterations 500 \
  --agent.seed 42 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.run-name r002-reproduction-seed42
```

### Option B: local TensorBoard logging

Use this when a W&B account is not available:

```bash
uv run train Mjlab-SpiRob-EggToBucket-Stage1 \
  --env.scene.num-envs 256 \
  --agent.max-iterations 500 \
  --agent.seed 42 \
  --agent.logger tensorboard \
  --agent.run-name r002-reproduction-seed42
```

The original R002 run processed 4,096,000 policy transitions in about 41.8
minutes on the RTX 3070 Ti Laptop GPU. Runtime will vary with the GPU, driver,
thermal limits, and other system load.

If GPU memory is insufficient, reduce only the parallel environment count for
the first attempt:

```bash
--env.scene.num-envs 64
```

Record that change. A 64-environment run is a new run, not an exact R002
reproduction.

## Play a trained checkpoint

List locally saved checkpoints:

```bash
find logs/rsl_rl/spirob_minimal -type f -name 'model_*.pt' -print
```

Then supply one exact checkpoint path:

```bash
uv run play Mjlab-SpiRob-EggToBucket-Stage1 \
  --agent trained \
  --checkpoint-file /absolute/path/to/model_checkpoint.pt \
  --num-envs 1 \
  --viewer auto
```

To play a checkpoint stored in W&B:

```bash
uv run play Mjlab-SpiRob-EggToBucket-Stage1 \
  --agent trained \
  --wandb-run-path ENTITY/mjlab/RUN_ID \
  --num-envs 1 \
  --viewer auto
```

Do not infer a policy's success probability from one visually convincing
playback. The present fixed scene can repeat nearly the same initial condition;
controlled multi-condition evaluation is a separate experiment.

## Export a W&B run for analysis

The repository includes a helper that exports configuration, scalar history,
summary data, diagnostic plots, and selected run files into a shareable ZIP:

```bash
uv run python tools/export_wandb_run_bundle.py \
  --run ENTITY/mjlab/RUN_ID \
  --output-root wandb_exports
```

The `--run` value may also be a full W&B run URL.

## Useful command discovery

The exact CLI is generated from the installed mjlab configuration. Check it
before adding or changing overrides:

```bash
uv run train Mjlab-SpiRob-EggToBucket-Stage1 --help
uv run play Mjlab-SpiRob-EggToBucket-Stage1 --help
```

For this pinned mjlab version, training environment count is
`--env.scene.num-envs`, whereas playback uses `--num-envs`.

## Repository layout

```text
spirob-mjlab/
├── src/spirob_mjlab/
│   ├── __init__.py          # Task registration
│   ├── env_cfgs.py          # Scene, observations, rewards, terminations
│   ├── actions.py           # Rate-limited tendon command
│   ├── mdp.py               # Observation/reward/termination functions
│   ├── entities.py          # Robot, egg, pedestal, bucket entities
│   ├── rl_cfg.py            # PPO configuration
│   └── assets/              # XML models and 21 SpiRob STL meshes
├── tools/
│   ├── validate_assets.py
│   ├── prepare_from_full_xml.py
│   └── export_wandb_run_bundle.py
├── docs/runs/R002.md        # First fixed-scene success record
├── pyproject.toml           # Direct dependency constraints
└── uv.lock                  # Reproducible Python environment
```

## Troubleshooting

### `uv run list-envs` does not show the SpiRob task

Run commands from the repository root and synchronize again:

```bash
uv sync --frozen
uv pip show spirob-mjlab
uv run list-envs | grep SpiRob
```

### `uv sync --frozen` says the lock file is stale

Do not regenerate the lock file for a baseline reproduction. Restore a clean
checkout and synchronize again:

```bash
git status --short
git switch main
git pull --ff-only
uv sync --frozen
```

### CUDA is unavailable

Check both layers:

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

If `nvidia-smi` fails inside WSL, fix the Windows NVIDIA driver/WSL GPU setup
before changing Python packages. If `nvidia-smi` succeeds but PyTorch reports
`False`, keep the repository lock file unchanged and record the full output of
`uv pip freeze` before troubleshooting dependencies.

### Native viewer does not open

Use:

```bash
uv run play Mjlab-SpiRob-EggToBucket-Stage1 \
  --agent zero \
  --num-envs 1 \
  --viewer viser
```

### W&B authentication fails

Either run `uv run wandb login` again or select local logging explicitly with:

```bash
--agent.logger tensorboard
```

### Training runs out of GPU memory

Reduce `--env.scene.num-envs` from 256 to 64 or 32. Do not simultaneously
change the reward, episode length, solver, action rate, or PPO settings; keeping
one controlled difference makes the run interpretable.

### NaN states appear

The task terminates environments with a non-finite acceleration state, but the
physical cause should still be investigated. Repeat a short diagnostic run
with:

```bash
uv run train Mjlab-SpiRob-EggToBucket-Stage1 \
  --env.scene.num-envs 64 \
  --agent.max-iterations 20 \
  --agent.seed 42 \
  --enable-nan-guard True \
  --agent.logger tensorboard \
  --agent.run-name nan-diagnostic
```

## Updating an existing installation

```bash
cd ~/Research_Work/spirob-mjlab
git switch main
git pull --ff-only
uv sync --frozen
uv run python tools/validate_assets.py
uv run list-envs | grep SpiRob
```

Always record the new commit hash before comparing a new run with an earlier
result.

## Development workflow

Do not develop directly on `main`. Create a branch from a recorded baseline:

```bash
git switch main
git pull --ff-only
git switch -c feature/short-purposeful-name
```

After making and checking one coherent change:

```bash
git status --short
git diff --check
git diff
git add README.md
git commit -m "docs: update setup instructions"
git push -u origin feature/short-purposeful-name
```

Open a pull request, record the validation run and its commit hash, and merge
only after the stated test passes.

## Scientific scope of Stage 1

Stage 1 asks one bounded question: can a fixed-base, two-tendon SpiRob learn to
move one egg from one known start position into one known bucket using object
progress and terminal outcomes?

The current answer is yes for the successful R002 run. The next scientific
steps are frozen-checkpoint evaluation, replication with independent training
seeds, and evaluation-only mapping of nearby egg start positions. Object-spawn
randomization should be introduced only after that fixed-policy map exists.
Base XY motion is not part of the present task and should not be added until
the fixed-base reachable workspace has been measured.

## References

- [mjlab documentation](https://mujocolab.github.io/mjlab/)
- [mjlab source repository](https://github.com/mujocolab/mjlab)
- [SpiRobs: Logarithmic Spiral-shaped Robots for Versatile Grasping Across Scales](https://arxiv.org/abs/2303.09861)
