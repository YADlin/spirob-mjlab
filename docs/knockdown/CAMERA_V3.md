# Rendered camera and hardware waypoint motion — v3

This branch implements NYK's 25 October choice: retain ordinary point-to-point
motion and use the selected common speed on both motors. A MuJoCo camera renders
the scene; the same black-egg detector and calibration arithmetic as the Lab
produce the actor's egg coordinates and visibility. The actor remains a small
seven-input MLP. No CNN, depth image, segmentation label or touch channel is
passed to it.

The original branches preserve the v2 checkpoint's original behavior. V3 changes
small-move speeds and the floor-success rule, so the old checkpoint is a candidate
to evaluate or fine-tune, not an automatically approved hardware policy.

## What is implemented

- `Mjlab-SpiRob-EggKnockdown-Camera`, `-Camera-Fixed`, `-Camera-Eval`: native
  MuJoCo-Warp RGB camera, 640×360, provisional 54.12-degree vertical FOV,
  camera at 321 mm. Its independent mocap body follows each environment origin,
  without following randomized robot droop.
- Shared `egg_vision.py` detects the black cap. Its largest-inscribed-disk centre
  avoids the centroid shift caused by an attached stem/shadow. The robot is
  rendered white, the egg black, pedestal white and floor brown.
- Shared `camera_geometry.py` uses 87 mm for the upright cap centre and 46 mm
  for the empty cup. It also scales each detector's expected area at that height.
- Image-only startup memory; lost detections hold the last seen coordinates and
  set visibility to zero. There is no simulator-truth fallback.
- Both cables use the selected common speed, continuously selectable from
  5–20 mm/s within the Lab session limit. D limits, move/HOLD/settle dispatch and
  1.2 s minimum decision period remain. Actual rig settling can take longer.
- Literal egg–floor contact is remembered across physics substeps. Unassisted
  falls are failures. Contacts are privileged reward/evaluation data only.
- Evaluation checks clearance at the actual requested placement, includes the
  full training rectangle, and supplies a genuine `hold` baseline. `zero`
  explicitly means D=0 and can move a curled robot.
- The exporter records the checkpoint's training contract separately, checks a
  passing camera evaluation, and fingerprints perception code for the Lab loader.

The original task names use geometric camera proxies; choose a **Camera** task
for the new image-driven workflow.

## Interactive playback fix (25 October)

Use the normal command to inspect movement before more training:

```bash
uv run play Mjlab-SpiRob-EggKnockdown-Camera --agent random
```

Previously, mjlab's viewer drew only the end of each 1.2-second decision. The
user's screen recording changes robot shape at approximately 1.2-second
intervals even though the desktop is recorded at 30 fps. The separate demo
video showed intermediate physics steps; the ordinary `play` command did not.

The project now supplies a `play` entry point that delegates to mjlab, with a
viewer extension for the three Camera tasks. It draws real integrated states
at 60 Hz (72 frames per decision) and paces them in wall time. The actor still
runs once per 1.2 seconds; all 600 physics steps, camera observations, rewards,
terminations, motor speeds and waypoint limits are unchanged. No interpolated
poses are invented. Native and Viser viewers use the extension; Viser's own
scene submission throttles its 60 Hz updates to about 30 Hz.

Expect a visible movement followed by the real HOLD pause. Random actions may
reverse direction and need not hit the egg. Smooth display is not evidence that
a policy succeeds or that physical transfer is validated. Episode resets still
jump to a new starting configuration. On a slow computer playback runs slower
than real time instead of skipping the motion. Pause and speed controls work
within a decision; reset/checkpoint changes apply at the next decision boundary.

The terminal should print `[SpiRob play] Drawing intermediate physics states`.
For the trained policy, replace `--agent random` with:

```bash
--checkpoint-file logs/rsl_rl/spirob_knockdown/kd-camera-v3-s2/model_99.pt
```

The headless regression uses the actual camera environment and native viewer
sync methods, substituting an EGL recorder only for the desktop window:

```bash
MUJOCO_GL=egl uv run python tools/knockdown/check_playback.py --out outputs/playback-check
```

It compares identical random actions with and without rendering, including
physics state, actor observations, rewards and done flags. This is a playback
regression, not a training run or policy acceptance test. The standard `--video`
recorder remains decision-rate; the fix here is interactive playback.

Validation: 47 camera/contract/playback tests pass. The native-viewer sync path
recorded four random decisions / 2,400 real physics steps / 289 frames (initial
frame plus 72 per decision). Final states, actor observations, rewards and done
flags exactly matched the same actions without the viewer (maximum error 0).
See `evidence/playback_fix/` for the clip, frame trace and summary. This test ran
on CPU with an EGL recorder replacing the desktop window; workstation CUDA and
the actual desktop window still need NYK's playback check. No hardware was run.

## Verified here

- Uploaded XML is byte-identical to training source, SHA-256
  `fe7d13cd476a453c2842747a405c31450de638643f67d92593e3a095c4f671da`.
- Uploaded `model_299.pt` SHA-256
  `094ffe13e4709397a6a68827e8a53e381e98e83a86b08dd577b733891f0a0eac`.
- 19 contract/physics tests and 23 camera/termination regression tests pass.
  The latter include 18 actual OpenGL-rendered placement checks with error <3 mm,
  hidden-egg behavior, image-only observation dependencies and contact precedence.
- Two-environment native camera reset and stepping produced (7,) actor and (63,)
  critic observations. One PPO update from the supplied checkpoint passed on CPU
  (4 transitions). This is pipeline validation, not a trained result.
- Frozen supplied policy: two nominal OpenGL-camera demonstrations succeeded,
  at (+80,110) mm / 10 mm/s and (−80,110) mm / 5 mm/s. Each took five decisions;
  actual floor contact occurred at 5.236 s and 5.436 s respectively.
- Native MuJoCo-Warp randomized camera smoke: 4/4 successes at 10 mm/s, no
  faults/falls/OOB/timeouts (four boundary placements on the negative-u side).
  Wilson 95% interval: 51.0–100%; this is not the 200-episode acceptance result.
- Curled-start hold baseline kept D=40 mm for all 30 decisions and timed out
  without robot–egg or egg–floor contact.
- Checkpoint → NumPy actor parity: maximum absolute error 1.05e-5 on 100
  observations, below 1e-4. All three shared Python files match the Lab byte for byte.
- Lab simulated-driver/runner checks: 37/37 passed (including the real dispatcher
  and motor Rig with DemoDrive), plus 9/9 offscreen RL-window checks. These are
  software tests, not physical trials.

See `evidence/camera_v3/` for demo summaries, per-decision observations/actions,
frames and training smoke metadata. No CUDA GPU or physical rig is available in
this workspace. Full randomized frozen evaluation and physical acceptance remain
pending. Camera calibration, lighting, distortion, occlusion and measured timing
still need comparison with the real camera; synthetic image accuracy is not
hardware accuracy. W&B's unprovided 471-byte training diff remains a provenance gap.

The GitHub connection was read-only for this write (HTTP 403). Import the supplied
bundles after extracting `SpiRob_camera_v3_handoff.zip` in `~/Downloads`. The
original branch heads remain untouched. Push the new branches from the workstation
if desired; they were not pushed by this session.

## Workstation: inspect the demo and evaluate the existing checkpoint

```bash
cd ~/Research_Work/spirob-mjlab
git fetch ~/Downloads/SpiRob_camera_v3_handoff/spirob-mjlab_camera-v3.bundle \
  feature/egg-knockdown-rendered-camera:feature/egg-knockdown-rendered-camera
git switch feature/egg-knockdown-rendered-camera
uv sync --frozen
export KD_CHECKPOINT="$PWD/logs/rsl_rl/spirob_knockdown/2026-10-24_16-57-12_kd-main-v2-s1/model_299.pt"
MUJOCO_GL=egl uv run python tools/knockdown/camera_demo.py \
  --checkpoint "$KD_CHECKPOINT" --out outputs/camera-v3-demo \
  --common-speed-mm-s 10 --video
uv run python tools/knockdown/evaluate.py \
  --checkpoint "$KD_CHECKPOINT" --task Mjlab-SpiRob-EggKnockdown-Camera-Eval \
  --episodes 200 --envs 16 --out outputs/camera-v3-eval --device cuda:0
```

The video is `outputs/camera-v3-demo/camera.mp4`. It renders intermediate physics
states at 30 fps while the policy still decides every 1.2 s. It shows real pauses;
rendering does not change the action period. On a desktop without EGL, omit
`MUJOCO_GL=egl` and use the normal MuJoCo OpenGL backend.

Keep each output directory new. Inspect `summary.json` and `episodes.csv`; do
not infer acceptance from the training reward. Target: >=190/200 successes,
with all faults/falls/timeouts retained. Repeat at the selected fixed hardware
speed with `--cable-speed-mm-s 5` (or the intended setting) before deploying at it.

## Training workflow from the next run: standard CLI and W&B

The completed `kd-camera-v3-s2` run used `tools/knockdown/train_camera.py` with
local TensorBoard logging. Its `model_99.pt` is compatible with the registered
Camera task. The helper remains for reproducing that historical run, but new
training uses the standard CLI below, at NYK's request. Do not restart training
merely to repair a viewer; check random playback first.

When continuing training from that checkpoint:

```bash
uv run train Mjlab-SpiRob-EggKnockdown-Camera \
  --env.scene.num-envs 32 \
  --agent.resume True \
  --agent.load-run kd-camera-v3-s2 \
  --agent.load-checkpoint model_99.pt \
  --agent.max-iterations 100 \
  --agent.seed 3 \
  --agent.algorithm.learning-rate 0.0001 \
  --agent.algorithm.schedule fixed \
  --agent.run-name kd-camera-v3-s3 \
  --agent.logger wandb \
  --agent.wandb-project mjlab
```

This resumes the learned networks, optimizer and saved iteration counter into
a new timestamped run under `logs/rsl_rl/spirob_knockdown/`; the terminal prints
the run folder and W&B URL. Unlike the historical helper, this uses mjlab's
standard continuation behavior, including randomized initial episode lengths.
The task chooses the environment; the checkpoint chooses the learned policy;
the run name identifies this training session in the logs and W&B. Evaluate
the newly produced checkpoint separately before export. Start with 32 image
environments; increase only after measuring GPU memory/throughput.

## Export and Lab

For the existing v2 checkpoint after it passes the v3 evaluation:

```bash
uv run python tools/knockdown/export_policy.py --checkpoint "$KD_CHECKPOINT" \
  --checkpoint-contract-id spirob-egg-knockdown/contract-v2 \
  --evaluation-summary outputs/camera-v3-eval/summary.json \
  --task Mjlab-SpiRob-EggKnockdown-Camera --out exports/camera-v3-candidate
```

For a fine-tuned checkpoint, use its path, passing evaluation summary and
`--checkpoint-contract-id spirob-egg-knockdown/contract-v3`. Do not use the
four-transition smoke checkpoint as a deployment candidate.

```bash
cd ~/Two_Cable_Motor_Setup/spirob-cable-studio
git fetch ~/Downloads/SpiRob_camera_v3_handoff/spirob-cable-studio_camera-v3.bundle \
  feature/rl-knockdown-camera-speed:feature/rl-knockdown-camera-speed
git switch feature/rl-knockdown-camera-speed
cd lab
sha256sum model/spirob_robot.xml
```

The hash must match the source above. Keep the calibrated model and camera
settings local. Follow `lab/docs/RL_KNOCKDOWN.md`: align the camera, sample the
black egg and empty white cup, verify coordinates at measured positions, then
load the export in **Dry run**. Real moves require the operator's existing
**Arm** control. Physical target: >=27/30 successes across at least 10 placements;
retain UNKNOWN/fault/manual labels. No motor firmware/WSL change is needed.
