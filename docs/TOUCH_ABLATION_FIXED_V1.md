# TCH-v1: Does binary touch feedback help SpiRob learn the fixed egg-to-bucket task?

**Status: protocol and implementation prepared; experimental training has not been run in this chat.** The first requested comparison is one pair (seed 42). Its report is exploratory. Six paired seeds (42–47) are the predeclared replication set; even six pairs cannot guarantee a decisive result.

## 1. Hypothesis and scope

With identical physics, initial conditions, state observations and PPO settings, access to the current 42 binary touch flags improves final deterministic egg-to-bucket performance and/or learning efficiency within 500 PPO updates.

This is a conditional question about the existing simulated task. It does not establish that physical tactile sensors are necessary, that force magnitudes help, or that a policy generalizes to another spawn or the corrected robot. All current non-touch observations remain available, including simulator-derived robot shape and egg-to-bucket displacement. Their information can make touch partly redundant in this task.

The physical egg, pedestal, bucket and robot reset states are fixed. There is no spatial grid, sector or spawn randomization in training or evaluation. PPO exploration, network initialization and the existing runner's initial episode-length staggering remain seed-controlled. The latter varies scheduling, not object spawn.

## 2. Single principal change

| Item | Touch | No touch |
|---|---|---|
| Task ID suffix | TouchAblation-Fixed-Touch | TouchAblation-Fixed-NoTouch |
| Touch input to actor and critic | Existing 42 thresholded flags | 42 constant zeros |
| Threshold | Raw reading greater than 1e-4 | Same raw computation, then masked |
| Actor input dimension | 93 | 93 |
| Critic input dimension | 93 | 93 |
| Hidden layers | 128, 128; ELU | Identical |
| XML touch sensors and collision geometry | Retained | Retained |
| Contact reward or tactile shaping | None added | None added |
| Start checkpoint | None; from scratch | None; from scratch |

The 93 inputs are 2 tendon lengths + 2 tendon velocities + 3 egg-to-bucket coordinates + 2 previous actions + 42 robot XY keypoints + 42 touch flags. The runtime check verifies dimensions and compiled sensor order before training. Masking both actor and critic avoids privileged tactile information in the no-touch learner. Keeping the same input dimensions permits exactly paired initial networks and removes architecture size as a confound. Observation normalizers start fresh for both arms; their later divergence is part of learning.

The raw sensor diagnostic is logged in both arms, never supplied separately to the actor, critic or reward. A touch reading need not identify egg contact specifically; the existing sensor implementation does not classify which object produced it. Because sensors remain active in both simulations, runtime measurements do not estimate the benefit of physically deleting sensors or disabling their computation.

## 3. Code, lockfile, seeds and checkpoints

- Repository: YADlin/spirob-mjlab.
- New branch: `research/touch-ablation-fixed-spawn-v1`.
- Frozen parent: `8e985fffdf60cd6c0ba453f053bc1d3b6cd9e654`, the source used by the recent legacy-model curriculum runs.
- Both tasks inherit only its Stage-1 fixed-scene configuration. They do not invoke the known incorrect legacy sector sampler.
- XML, meshes, stiffness, damping, collision parameters, action mapping, rewards, terminations and `uv.lock` remain unchanged.
- Paired seeds: 42, 43, 44, 45, 46, 47. Seed 42 is the initial pair. Odd seeds run no-touch first; even seeds run touch first.
- Each new seed initializes both actor and critic from scratch. Initial model state hashes and parameter counts must match within a seed pair.
- Each run records source commit, lockfile hash, commands, initial model hashes, elapsed training time, checkpoint paths and SHA-256 values. Every evaluation records task-package hash and package versions.
- Final checkpoint is fixed at `model_499.pt`; no best-of-run selection and no success-dependent extension of one arm.

Different source versions or corrected physics must be treated as a separate experiment. Do not merge other chats' changes into this branch midway through a pair. Existing checkpoints and ongoing model/sector corrections retain their own records.

## 4. Training distribution and matched budget

| Control | Frozen value |
|---|---|
| Egg initial position, environment-local XYZ | (0.05, 0.15, 0.098) m |
| Pedestal entity position | (0.05, 0.15, 0.002) m |
| Bucket entity position | (-0.05, 0.15, 0.0) m |
| Episode horizon | 8 s |
| Physics timestep / decimation | 0.0001 s / 100 |
| Policy period | 0.01 s |
| Parallel training environments | 256 |
| Rollout steps per environment per update | 32 |
| PPO updates per run | 500 |
| Training transitions per arm | 4,096,000 |
| Total transitions for the initial pair | 8,192,000 |
| Total transitions for six pairs | 49,152,000 |

The solver and action-rate limits are inherited without retuning. A low-performing seed remains evidence. A crashed/incomplete training run must be reported as incomplete, never replaced silently by a successful seed. Hardware and software execution details must accompany the result; wall-clock time is a secondary, machine-dependent measure.

## 5. Frozen evaluation distribution

Evaluate saved checkpoints `model_0.pt`, `model_100.pt`, `model_200.pt`, `model_300.pt`, `model_400.pt` and `model_499.pt`. Under the pinned from-scratch runner these represent 1, 101, 201, 301, 401 and 500 completed PPO updates. `model_0.pt` is already trained for one update; it is not an untrained baseline.

At each intermediate checkpoint run five deterministic nominal execution repeats, and at the final checkpoint run twenty. Use one simulator environment, seeds 3210 onward in repeat order, and the exact same reset condition. No action sampling is used during evaluation. `auto_reset=False` preserves the terminal state so a reset cannot be mistaken for the last state of an episode. Raw flags are read from the existing success/fall/OOB/NaN/timeout terms. Only an unambiguous success contributes to the success numerator; every completed episode stays in the denominator. Nonfinite sensors and overlapping outcomes are reported for review.

These repeats measure computational repeatability at one physical condition. They are not twenty distinct spawns or twenty independent training runs. More identical episodes cannot establish workspace coverage. The independent replication unit for the touch/no-touch comparison is the training seed.

Each episode CSV includes seed, checkpoint/source hashes, reset coordinates, outcome, return, duration, terminal state, trajectory fingerprint and raw touch activation counts. The first episode of every evaluated checkpoint also has a trace of touch activation, egg position and commanded policy actions.

## 6. Analysis, tables and figures

For seed s, let p_touch,s and p_no-touch,s be final success fractions across the twenty fixed-condition repeats. The primary paired effect is d_s = p_touch,s − p_no-touch,s; report the mean d_s in percentage points. Report each seed, not just a pooled average.

| Output | Meaning and limitation |
|---|---|
| Frozen-policy success versus training transitions | Per-seed curves and condition means; distinguishes learning progress from noisy training reward |
| Paired final-success plot and seed table | Connects the two conditions sharing one training seed; no pseudoreplication from execution repeats |
| Outcome composition | Success, fall, timeout, OOB, NaN and other/overlap; no failed-episode deletion |
| Raw touch activation by channel | Establishes whether contact information is available; activity alone does not prove causal use by the policy |
| Learning-curve area (AUC) | Trapezoidal area normalized over the sampled window from 8,192 to 4,096,000 transitions; secondary learning-efficiency measure |
| Completion time among successes | Read alongside success fraction; conditioning on success can bias comparisons |
| Training wall time | Includes startup; same machine needed for interpretation; does not measure sensor-computation removal |

Graphs are exported as PNG and PDF; exact tables as CSV; the narrative as Markdown plus machine-readable JSON. No empirical graph or result will be populated before evidence exists. Training W&B reward and termination-event telemetry are supporting diagnostics, never substituted for frozen success fractions.

### Predeclared interpretation

A five-percentage-point absolute change in final success is the proposed practical margin for this task. It is a protocol choice, not a universal robotics standard. Do not change it after seeing the runs.

- With one pair, report what happened and mark the conclusion exploratory.
- With all six pairs, report an approximate 95% paired Student-t interval over the six seed differences and a two-sided exhaustive sign-flip p-value. The t interval assumes a suitable sampling model for seed differences; the sign-flip test assumes exchangeability/symmetry under the null. Both assumptions and the small sample size limit inference.
- A practical final-performance benefit requires the interval to lie above +5 percentage points and sign-flip p < 0.05, with interpretable engineering outcomes. The reverse supports harm under these constraints.
- An interval entirely within -5 to +5 points supports practical equivalence of final performance under the assumed model; learning efficiency is assessed separately.
- An interval crossing these bounds is inconclusive. If all seed differences are identical, the script does not report a misleading zero-width uncertainty interval; inspect ceiling/floor effects and exact outcomes.
- Similar final performance with consistently higher AUC supports a limited learning-efficiency finding, not final-performance superiority. AUC is secondary and does not override the preregistered primary endpoint.
- If neither arm learns, the experiment may be limited by exploration, objective or budget; this does not establish that touch is useless. If both reliably succeed early, ceiling effects can hide a benefit.

Six seeds are a feasible first replication set, not a power calculation or a promise of conclusiveness. Do not extend training selectively or stop collecting seeds once significance appears. Any further replication requires a newly recorded fixed protocol.

## 7. Exact run sequence and next evidence

After importing the delivered branch, run the initial pair from its isolated worktree:

```bash
cd /home/yadu/Research_Work/spirob-touch-ablation
uv sync --frozen
uv run python tools/run_touch_ablation.py \
  --output-root /home/yadu/Research_Work/spirob-evidence/touch_ablation_v1 \
  --seeds 42
```

The command runs a live preflight, trains touch and no-touch separately, evaluates all six checkpoints for each, verifies paired initial model hashes, and creates the report. It refuses to overwrite an existing training run. With completed training and pending evaluation, use the same command with `--evaluate-only`; incomplete evaluation folders remain visible and must be inspected before rerun.

For the predeclared replication set, once the initial pair is operationally validated:

```bash
uv run python tools/run_touch_ablation.py \
  --output-root /home/yadu/Research_Work/spirob-evidence/touch_ablation_v1 \
  --seeds 43 44 45 46 47
```

This adds five pairs and reports all six, retaining seed 42. No warm-start checkpoint is supplied to either arm. If W&B is unavailable, select `--logger tensorboard` consistently for all runs.

Send the generated report directory, all `run.json` manifests, and evaluation CSV/JSON/trace files. Keep the full training logs and checkpoint files locally. The result can then be verified and inserted into the Notion research page, with actual figures and a bounded conclusion.

## References and source anchors

- `src/spirob_mjlab/mdp.py`: `touch_values`, `robot_keypoints_xy`, existing task outcomes.
- `src/spirob_mjlab/env_cfgs.py`: Stage-1 observations, actions, reward, solver and horizon.
- `src/spirob_mjlab/rl_cfg.py`: PPO architecture and optimization configuration.
- Pinned MJLab 1.4.0 `scripts/train.py` and RSL-RL 5.2.0 runner: initialization, episode-length staggering and checkpoint indexing.
- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., and Bellemare, M. G. (2021), *Deep Reinforcement Learning at the Edge of the Statistical Precipice*, NeurIPS. [Author manuscript](https://arxiv.org/abs/2108.13264). Supports reporting uncertainty and seed variability; the present single-task protocol and practical margin are our experiment-specific choices.
