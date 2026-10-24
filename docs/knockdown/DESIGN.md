# Egg knockdown — design note (contract v2)

**Bottom line.** The policy moves the SpiRob with **one number, the cable difference
D = L0 − L1**, while the cable mean stays at 227.35 mm. It sees only what the rig can
see: the egg position from the overhead camera (with a visible/not-visible flag) and
D from the motor encoders. Each decision is one point-to-point move of at most 8 mm
in D. The move speed is capped by an operator-selected ceiling (5, 10 or 15 mm/s) and
scaled to the move size, so small moves are slow, never sudden.

Checked by `tests/test_knockdown_contract.py` (19 tests) and the Lab's
`tests/verify_rl_knockdown.py` (37 checks, closed loop through a MuJoCo twin).

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
| 0–1 | egg (u, v) | camera; last value held while not visible | true egg xy + bias (±3 mm/episode) + noise (1 mm) | /0.10 m |
| 2 | egg visible | detector found it this frame | inside the camera rectangle | 1 |
| 3–4 | egg (u, v) at episode start | camera, latched | same estimator at reset | /0.10 m |
| 5 | D | encoders: payout0 − payout1 | simulated motor positions | /0.080 m |
| 6 | previous action | policy memory | same | 1 |

Not in the actor: touch, force, tension, joint angles, link positions, egg height.
The critic gets 63 privileged numbers (true egg pose and velocity, link positions,
tendon lengths, contact flag, robot drop, speed ceiling). It is not exported.

Frame C: origin at joint `link_002` at rest; u = model +x, v = model −y (base → tip).
Base point (link_001) is at v = −12 mm. **D > 0 curls the robot toward −u.**

## 4. What the action means

```
raw a ─clip[-1,1]─► desired D = 80 mm·a ─step ≤ 8 mm─► limited D ─|ΔD| < 0.5 mm? → no move
                                                         │
             per-cable speed = clip(vmax · |ΔD| / 8 mm, 0.5 mm/s, vmax)
                                                         │
   rig:  Lab 'independent' move to L0 = 227.35 + D/2, L1 = 227.35 − D/2 at that speed; wait for HOLD
   sim:  motor positions ramp at that speed; tendon target = payout + offset (±2 mm per episode)
```

**Worked example (vmax = 15 mm/s).** A full 8 mm step in D moves each cable 4 mm
at 15 mm/s → 0.27 s. A 1 mm step moves each cable 0.5 mm at 15/8 = 1.9 mm/s →
also 0.27 s. Every move takes the same time, so a tiny correction is never a jerk.
At vmax = 5 mm/s every move takes 0.8 s, inside the 1.2 s decision.

* Decision period 1.2 s (physics 2 ms × 600). 30 decisions per episode (36 s).
* Training samples vmax in [4.5, 15] mm/s, so one policy works at all three settings.
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
| Held-out evaluation | 50–130 mm | 60–170 mm |
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

Termination order: numerical fault › knocked down (egg below cup top − 5 mm and
touched by the robot) › spontaneous fall › egg out of bounds › time-out.

**Rig success** (camera only), either:
1. egg seen > 35 mm from its start for ≥ 0.5 s, or
2. the empty blue cup is seen at the start position with no egg within 35 mm for ≥ 0.7 s
   (covers an egg knocked out of frame).

## 7. Human-in-the-loop on the rig (auto-continue)

After a success the robot stays where it is. The operator puts the egg back on the
pedestal somewhere new. The next episode starts by itself once the scene has been
still for 2 s (hand gone), the egg has been stable within 2 mm, and it is > 25 mm from
the robot. In training this is simply an episode reset from a random curled shape.

## 8. Colours

| Object | Colour | Sim rgba | Hex |
|---|---|---|---|
| Robot | white (as built) | — | — |
| Floor | cardboard brown | — | — |
| Egg | saturated green | 0.16 0.67 0.31 | #29AB4F |
| Pedestal | saturated blue | 0.12 0.35 0.78 | #1F59C7 |

Green and blue are far in hue from brown and far in saturation from white; avoid
red, orange or yellow (close to cardboard).

## 9. Commands

See the README section "Egg knockdown" and `lab/docs/RL_KNOCKDOWN.md` in
spirob-cable-studio. The v1 note and reach map are kept in `docs/knockdown/v1/`.
