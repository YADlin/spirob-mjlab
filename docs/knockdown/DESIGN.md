# Egg knockdown — design note (contract v3)

Current commands and validation: [rendered camera workflow](CAMERA_V3.md).

The actor uses seven camera/encoder/memory values and selects a cable difference
D = L0 − L1 at fixed mean 227.35 mm. Each waypoint changes D by at most 8 mm.
Both cables use the selected common speed (5–20 mm/s), including small moves.
The motor owner waits for HOLD before accepting another target. This protocol is
unchanged; no mid-motion retargeting or extra motion smoothing is introduced.

The `-Camera` tasks render real MuJoCo-Warp RGB frames and run the shared Lab
black-egg detector. The original task names remain geometric-proxy baselines.
Contract v2 weights require evaluation under v3 before use; tensor compatibility
alone does not establish behavioral compatibility.

Drawings: [rig_geometry.png](rig_geometry.png) (camera and robot heights) and
[egg_pedestal.png](egg_pedestal.png) (parts to print). Reach map: [reach_map_v2.png](reach_map_v2.png).

## 1. Robot model and where it came from

| | v1 (old) | v2 (this task) |
|---|---|---|
| File | `assets/lab_model/` (removed) | `assets/knockdown/source/spirob_robot_k2_2.148_20261024_0318.xml` (verbatim, sha256 `fe7d13cd476a…`) |
| Used by the env | — | `assets/knockdown/spirob_robot.xml` = source minus floor plane and 3 lights (`tools/knockdown/make_knockdown_assets.py`) |
| Joint 2 stiffness | 0.183 | 2.148 (11.7× stiffer) |
| Mass, body positions, hulls, names | same | same |
| Agrees with the real robot | — | mean ≈ 227.4 mm, D from −80 to +80 mm (NYK) |

The egg-to-bucket assets (`assets/spirob_robot.xml`, `spirob_egg.xml`,
`spirob_pedestal.xml`) are untouched; a test checks their hashes. The Lab's
`lab/model/spirob_robot.xml` must be this same source file (the policy export
records its full hash and the Lab refuses a mismatch).

## 2. Geometry (heights from the table, mm)

```
camera lens ─────────────── 321      camera centre is 140 mm from the base point,
                                     horizontally, above the robot's rest axis (to confirm)
egg top ─────────────────── 107
model robot plane ───────── 90       real robot: base top 88, tip top ≈ 70 (droop)
egg cap centre ──────────── 87
pedestal cup top ────────── 46       lowered 20 mm from the original 66
table ───────────────────── 0
```

**Why the pedestal was lowered.** With the original 66 mm cup top, the drooping
real tip (≈ 70 mm top, so about 60 mm at its underside) would sweep into the cup
instead of the egg. At 46 mm the lowest robot point stays ≥ 12 mm above the cup and
the robot meets the 40 mm egg cap (67–107 mm). Training also lowers the whole model
robot by a random 0–15 mm per episode to cover the droop.

## 3. What the actor sees (7 numbers)

| # | Term | On the rig | In simulation | Scale |
|---|---|---|---|---|
| 0–1 | egg (u, v) | camera; last value held while not visible | rendered RGB → black-egg detector → calibrated coordinates (`-Camera` tasks) | /0.10 m |
| 2 | egg visible | detector found it this frame | detector found a component in this image | 1 |
| 3–4 | egg (u, v) at episode start | camera, latched | same estimator at reset | /0.10 m |
| 5 | D | encoders: payout0 − payout1 | simulated motor positions | /0.080 m |
| 6 | previous action | policy memory | same | 1 |

Not in the actor: touch, force, tension, joint angles, link positions, egg height.
The critic gets 63 privileged numbers (true egg pose and velocity, link positions,
tendon lengths, contact flag, robot drop, common speed). It is not exported.

Frame C: origin at joint `link_002` at rest; u = model +x, v = model −y (base → tip).
Base point (link_001) is at v = −12 mm. **D > 0 curls the robot toward −u.**

## 4. What the action means

```
raw a ─clip[-1,1]─► desired D = 80 mm·a ─step ≤ 8 mm─► limited D ─|ΔD| < 0.5 mm? → no move
                                                         │
             per-cable speed = operator-selected common speed
                                                         │
   rig:  Lab 'independent' move to L0 = 227.35 + D/2, L1 = 227.35 − D/2 at that speed; wait for HOLD
   sim:  motor positions ramp at that speed; tendon target = payout + offset (±2 mm per episode)
```

**Example at 15 mm/s.** An 8 mm D step moves each cable 4 mm in 0.267 s.
A 1 mm D step moves each cable 0.5 mm in 0.033 s. Both then hold.

* Minimum decision period 1.2 s (physics 2 ms × 600); 30 decisions per episode.
* Training samples common speed in [5, 20] mm/s. Hardware session limits still apply.
* Actual hardware also waits for settled encoder telemetry and can take longer.
* The episode can start at any D: in training 70 % of episodes start from a random
  curled shape taken from the pose table (`pose_table.json`, D every 2.5 mm), which is
  what happens on the rig when the robot stops after a knockdown.

## 5. Where the egg can be placed

Scripted D sweeps ±80 mm at 5 mm/s per cable (`studies/reach_map_v2.py`):
reliable knockdowns at v ≈ 6–18 cm and |u| ≈ 4–14 cm on both sides; the robot at
rest overlaps |u| ≤ 2 cm. A 12 mm robot drop changes almost nothing. The median |D|
at the moment the egg falls grows from 15 mm (|u| = 4 cm) to about 50 mm (|u| ≥ 10 cm).

| Region | abs u | v |
|---|---|---|
| Training | 45–140 mm | 50–180 mm |
| Held-out evaluation | 45–140 mm | 50–180 mm |
| Fixed smoke task | u = 80, v = 110 | |

Every placement must clear the robot's start shape by > 25 mm (same rule on the rig).
Hardware placements P01–P12 lie inside the evaluation region.

**Camera view (provisional).** Centre (u, v) = (0, 128) mm, half-size 200 × 118 mm.
Replace with the measured view after the camera re-alignment. Outside it the egg is
"not visible" and the actor keeps the last seen position.

## 6. Reward, outcomes and success

| Term | Weight | Notes |
|---|---|---|
| knockdown | +10 once | episode ends |
| approach progress | 1.0 | Φ = −min(link–egg distance, 10 cm)/10 cm; reward ΔΦ |
| egg displacement progress | 2.0 | Φ = min(egg offset, 3 cm)/3 cm; reward ΔΦ |
| commanded motion | −0.02 | per full 8 mm move |
| time | −0.04 | per decision |

Termination order: numerical fault › actual egg–floor contact (success only after
robot contact; otherwise spontaneous fall) › egg out of bounds › time-out. Contact
history is updated each physics substep and survives separation until reset.
A floor success or fault is never simultaneously marked as a time-limit truncation.

**Rig success** (camera only), either:
1. egg seen > 35 mm from its start for ≥ 0.5 s, or
2. the empty white cup is seen at the start position with no egg within 35 mm for ≥ 0.7 s
   (covers an egg knocked out of frame).

## 7. Human-in-the-loop on the rig (auto-continue)

After a success the robot stays where it is. The operator puts the egg back on the
pedestal somewhere new. The next episode starts by itself once the scene has been
still for 2 s (hand gone), the egg has been stable within 2 mm, and it is > 25 mm from
the robot. In training this is simply an episode reset from a random curled shape.

## 8. Colours (only white and black PLA available)

| Object | Colour | Sim rgba |
|---|---|---|
| Robot | white (as built) | — |
| Floor | cardboard brown | — |
| Egg | **black PLA** (matte if possible) | 0.06 0.06 0.06 |
| Pedestal | **white PLA** | 0.93 0.93 0.93 |

Why this way round: the egg is the object tracked every frame, so it must stay a
separate blob when the white robot touches it; black does, white would merge with the
robot. The empty cup must never look like the egg; a white cup cannot be mistaken for
a black egg (a black cup could, and success would then never be detected). The one
look-alike for a black egg is a hard robot shadow, so use soft overhead light; the
detector also rejects blobs of the wrong size or shape (Lab test: robot + shadow +
empty cup in view → no egg reported). The policy never sees colours; they matter only
for the shared detector in both rendered training and the Lab.

## 9. Commands

See the README section "Egg knockdown" and `lab/docs/RL_KNOCKDOWN.md` in
spirob-cable-studio. The v1 note and reach map are kept in `docs/knockdown/v1/`.
