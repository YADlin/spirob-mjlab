# Egg knockdown — design note (contract v1)

**Bottom line.** The knockdown task trains on the *current Lab robot model* with an
actor that sees only what the rig can see: the camera's egg position and the motor
encoders. Each policy decision is one bounded point-to-point cable move, because
that is the only motion the Lab's motor owner accepts. Everything in this file is
checked by `tests/test_knockdown_contract.py` (fast) and the Lab's
`tests/verify_rl_knockdown.py` (closed loop through a MuJoCo twin).

## 1. What the actor sees (8 numbers)

| # | Term | Source on the rig | Simulation stand-in | Scale |
|---|---|---|---|---|
| 0–1 | egg position (u, v) | overhead camera | true egg XY + per-episode bias (±3 mm) + noise (1 mm) | /0.10 m |
| 2–3 | egg position at episode start | camera, latched | same estimator at reset | /0.10 m |
| 4–5 | motor payout − neutral | encoders | simulated motor position | /0.060 m |
| 6–7 | previous action | policy memory | same | ×1 |

Not in the actor: touch, force, tension, joint angles, simulator backbone points,
egg height. The Lab model has **zero sensors**, so touch data cannot leak. The
critic gets privileged state (true egg pose/velocity, 21 link positions, tendon
lengths, contact flag, reference offset); it is not exported.

Frame C is the Lab's calibrated plane frame: origin at joint `link_002` at rest,
u = model +x, v = model −y (base → tip). The Lab camera alignment already maps
frame C to pixels, so no new frame is introduced.

## 2. What the action means

```
raw a ──clip[-1,1]──► desired = neutral + 60 mm · a ──step limit ±8 mm──► limited
       └─ hardware: Lab 'independent' move to limited at 15 mm/s, wait until both axes HOLD
       └─ sim:      motor position ramps to limited at 15 mm/s (×U[0.85,1.05]); tendon target = payout + offset(±2 mm)
```

* One decision = 1.0 s (provisional; measure on the rig). An 8 mm move at 15 mm/s
  takes 0.53 s, so every move finishes inside a decision even at the slow end of the
  speed randomisation.
* **Rate mismatch resolved without raising limits:** Stage-1 allowed 1 mm/10 ms =
  100 mm/s; the Lab ceiling is 20 mm/s. The knockdown contract uses 15 mm/s.
* Raw action, desired target, limited command and measured payout are separate
  fields in every hardware log row.
* Encoder payout is the motor's position, not the geometric tendon length: when one
  cable is paid out the robot does not follow it exactly (slack). The simulator
  therefore drives the tendon *target* from the payout and the actor sees payout.

## 3. Physics settings and why

| Setting | Value | Evidence |
|---|---|---|
| Robot | Lab model (hash `ac8d9f9d…`), floor plane removed | entity test: same bodies/joints/tendons/mass, 0 sensors |
| Physics step | 2 ms, implicitfast | free motion: tip deviation ≤ 0.055 mm vs 0.1 ms; knockdown time within 0.03 s of 1 ms at 6 placements |
| Decimation | 500 (1.0 s decisions) | contract decision period |
| Egg at rest | z = 95.0 mm on the cup | 5 s with no action: drift 0.02 mm (also checked at 4 placements) |

Note: the Lab model differs from the older RL robot (mass 53 g vs 23 g, mesh-hull
collisions, ±180° joint ranges, base joint k = 100 vs 5000, neutral 227.35 vs 219.8 mm).
The older robot is not used by this task.

## 4. Where the egg can be placed

Scripted full differential curls at 20 mm/s on a 2 cm grid (`studies/reach_map.py`,
`reach_map.png`): reliable knockdowns for v ≈ 6–18 cm and |u| ≈ 4–14 cm on both
sides; the robot only grazes the egg near the mount (v ≤ 4 cm) and overlaps it at
rest for |u| ≤ 4 cm.

* Training region: |u| ∈ [4.5, 10.5] cm, v ∈ [5.5, 17.5] cm, both sides.
* Held-out evaluation region: |u| ∈ [5, 10] cm, v ∈ [6, 17] cm (boundaries included).
* Hardware placements P01–P12 (`lab/rl/placements.py`) lie inside the evaluation region.
* Egg and pedestal move together; pedestal is a mocap prop fixed during the episode.

## 5. Reward and outcomes

| Term | Weight | Notes |
|---|---|---|
| knockdown | +10 once | episode ends on the same condition |
| approach progress | 1.0 | Φ = −min(link–egg distance, 10 cm)/10 cm, reward ΔΦ (telescopes; cannot be farmed) |
| egg displacement progress | 2.0 | Φ = min(egg offset from pedestal, 3 cm)/3 cm, reward ΔΦ |
| commanded motion | −0.02 | per full 8 mm move on both cables |
| time | −0.05 | per decision |

Termination precedence (mutually exclusive): numerical fault › knocked down (egg
below cup top − 5 mm **and** robot touched it this episode) › spontaneous fall (fell
without contact) › egg out of bounds › 24-decision time-out.

On hardware, success = egg detected > 35 mm from its start for ≥ 0.5 s. A lost or
occluded egg is UNKNOWN (operator labels it from the recorded video), never success.

## 6. Commands

See the repository README section "Egg knockdown" for training, evaluation and
export commands, and `lab/docs/RL_KNOCKDOWN.md` in spirob-cable-studio for the rig.
