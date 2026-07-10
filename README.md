# SpiRob mjlab minimal task

This is a deliberately small mjlab task for confidence-building.

It registers one task:

```text
Mjlab-SpiRob-Minimal
```

The scene contains all required entities:

- `robot`: fixed-base tendon-actuated SpiRob, loaded from `spirob_robot.xml`
- `pedestal`: fixed pedestal under the object
- `egg`: free object on the pedestal
- `bucket`: fixed bucket/drop target
- `terrain`: mjlab plane terrain

The first task is not full pick-and-place. It is a minimal reach/contact task:

- action: 2 tendon-length commands for `cable_0` and `cable_1`
- observation: tendon lengths, tendon velocities, touch sensors, tip-to-egg vector, egg-to-bucket vector, last action
- reward: tip approaches egg, touch sensors activate, small action penalty
- termination: touch/near egg success, egg falls, NaN, timeout

This is intended to prove that mjlab can instantiate your robot and run 64 parallel envs before adding harder manipulation logic.

---

## 1. Required asset placement

Your `spirob_robot.xml` references STL meshes:

```text
src/spirob_mjlab/assets/meshes/link_001.stl
...
src/spirob_mjlab/assets/meshes/link_021.stl
```

Copy your existing mesh files into:

```text
spirob_mjlab_minimal/src/spirob_mjlab/assets/meshes/
```

Then validate the files:

```bash
cd spirob_mjlab_minimal
python tools/validate_assets.py
```

If this says meshes are missing, MuJoCo/mjlab will not compile the robot.

---

## 2. Install inside your mjlab environment

Recommended WSL layout:

```bash
cd ~
git clone https://github.com/mujocolab/mjlab.git
cd mjlab
uv sync
```

Then install this task package editable:

```bash
uv pip install -e /path/to/spirob_mjlab_minimal
```

Check that mjlab sees the task:

```bash
uv run list-envs | grep SpiRob
```

Expected task id:

```text
Mjlab-SpiRob-Minimal
```

---

## 3. Sanity checks

Start with zero and random policies before PPO:

```bash
uv run play Mjlab-SpiRob-Minimal --agent zero --env.scene.num-envs 1
uv run play Mjlab-SpiRob-Minimal --agent random --env.scene.num-envs 1
```

If these fail, fix XML/mesh/import issues before training.

---

## 4. Train 64 environments

```bash
uv run train Mjlab-SpiRob-Minimal \
  --env.scene.num-envs 64 \
  --agent.max-iterations 200
```

For a first confidence run, 100 to 200 iterations is enough to see whether the reward increases and whether contacts occur. After this works, increase iterations and add staged manipulation rewards.

---

## 5. What to build next

Stage 0: this package, all entities present, reach/contact only.

Stage 1: keep contact and move egg in XY.

Stage 2: contact-maintaining carry reward using egg XY displacement.

Stage 3: bucket delivery success condition.

Stage 4: randomize egg/bucket positions and mesh/friction parameters.
