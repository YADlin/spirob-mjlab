# SpiRob mjlab — Blank-System Setup for Collaborator

This guide assumes a fresh Windows computer and takes the user from zero to running the professor-facing SpiRob mjlab checkpoint.

> IMPORTANT  
> Do not use the latest mjlab `main` branch unless explicitly instructed. Use the exact mjlab commit tested by the SpiRob repository maintainer.

## What you need from the SpiRob researcher

Before beginning, obtain:

1. **SpiRob Git repository URL**
2. **Branch:** `prof-share-stage1`
3. **Exact tested mjlab commit:** `0cdc56246999409b83622764f5b38edb660cf16e`
4. Optional demonstration checkpoint: `<CHECKPOINT_FILE_OR_RELEASE_URL>`

The SpiRob repository contains the robot XMLs, meshes, custom mjlab task package, actions, observations, rewards, terminations, PPO configuration, and documentation.

## A. Hardware check

For GPU training, the system should have an NVIDIA CUDA-capable GPU.

On Windows PowerShell:

```powershell
nvidia-smi
```

If this command is unavailable, install/update the appropriate NVIDIA Windows driver before attempting GPU training.

## B. Install WSL2 + Ubuntu

Open **PowerShell as Administrator**:

```powershell
wsl --install
```

Restart Windows if requested.

After restart, launch Ubuntu and create the Linux username/password when prompted.

Update WSL from PowerShell:

```powershell
wsl --update
```

Enter Ubuntu:

```powershell
wsl
```

Inside Ubuntu, verify GPU visibility:

```bash
nvidia-smi
```

Do not install a separate NVIDIA Linux display driver inside WSL. GPU access is provided through the Windows NVIDIA driver.

## C. Install basic Linux tools

Inside Ubuntu:

```bash
sudo apt update
sudo apt install -y git curl build-essential
```

## D. Install uv

Inside Ubuntu:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the shell, or run:

```bash
source ~/.bashrc
```

Check:

```bash
uv --version
```

## E. Create the working directory

```bash
mkdir -p ~/Research_Work
cd ~/Research_Work
```

## F. Clone and pin mjlab

```bash
git clone https://github.com/mujocolab/mjlab.git
cd mjlab
```

Check out the exact commit provided by the researcher:

```bash
git checkout <MJLAB_COMMIT_HASH>
```

Verify:

```bash
git rev-parse HEAD
```

Install/synchronize mjlab:

```bash
uv sync
```

Verify mjlab itself:

```bash
uv run demo
```

If this fails, stop and fix the mjlab/GPU installation before installing SpiRob.

## G. Clone the SpiRob repository

```bash
cd ~/Research_Work
git clone https://github.com/YADlin/spirob-mjlab spirob-mjlab
cd spirob-mjlab
git switch prof-share-stage1
```

Verify:

```bash
git branch --show-current
```

Expected:

```text
prof-share-stage1
```

## H. Install the SpiRob package into the mjlab environment

```bash
cd ~/Research_Work/mjlab
uv pip install -e ~/Research_Work/spirob-mjlab
```

Check:

```bash
uv pip show spirob-mjlab
```

## I. Verify task registration

```bash
uv run list-envs | grep SpiRob
```

Expected tasks include:

```text
Mjlab-SpiRob-Minimal
Mjlab-SpiRob-BucketDrop
Mjlab-SpiRob-EggToBucket-Stage1
```

## J. First SpiRob test — zero policy

```bash
uv run play Mjlab-SpiRob-EggToBucket-Stage1   --agent zero   --num-envs 1
```

Expected result:

- viewer opens;
- robot, pedestal, egg and bucket are visible;
- simulation runs without immediate crash.

## K. Second test — random policy

```bash
uv run play Mjlab-SpiRob-EggToBucket-Stage1   --agent random   --num-envs 1
```

Expected result:

- tendon actions move SpiRob;
- physics remains numerically stable;
- object contact can occur.

## L. Optional demonstration checkpoint

If a trained policy checkpoint has been provided, inspect the installed mjlab play syntax:

```bash
uv run play Mjlab-SpiRob-EggToBucket-Stage1 --help
```

Then use the checkpoint argument documented by that installed mjlab version.

The provided checkpoint demonstrates a research milestone, not necessarily complete egg-to-bucket transfer.

## M. Optional training test

Only after the zero/random tests work:

```bash
uv run train Mjlab-SpiRob-EggToBucket-Stage1   --env.scene.num-envs 64   --agent.max-iterations 10
```

This is only a pipeline check.

## N. What the current checkpoint demonstrates

The professor-facing branch represents a working research implementation, not a completed manipulation solution.

Verified capabilities include:

- mjlab task registration;
- vectorized GPU simulation;
- SpiRob tendon actuation;
- rate-limited tendon commands;
- robot, pedestal, egg and bucket entities;
- touch-sensor observations;
- corrected vectorized environment coordinates;
- reward and termination pipeline;
- learned approach/contact/curling behaviours.

Current research is progressing from one-sided contact toward physically meaningful capture/enclosure and eventual directed transport to the bucket.

## O. Troubleshooting information to send back

If setup fails, send the researcher the output of:

```bash
nvidia-smi
uv --version
python --version
git -C ~/Research_Work/mjlab rev-parse HEAD
git -C ~/Research_Work/spirob-mjlab rev-parse HEAD
uv pip show mjlab
uv pip show spirob-mjlab
uv run list-envs | grep SpiRob
```

Also include the complete traceback from the failed command.

## P. Reproducibility rule

For research comparison:

- use the specified SpiRob branch/tag;
- use the specified mjlab commit;
- do not edit XML/reward/action parameters before reproducing the reference behavior;
- record the exact command and checkpoint used.
