# TCH-v1 implementation validation — 10 September 2026

These are software checks, not evidence that touch helps or harms learning.

- Live preflight on CPU with MuJoCo 3.8.1, MuJoCo Warp 3.8.1, Warp 1.14.0, MJLab 1.4.0: passed.
- Configurations compared structurally, including callable definitions: the ablation arms differ only in touch visibility. Scene, resets, actions, rewards, terminations, physics and decimation match the frozen Stage-1 source.
- Compiled sensor identity/order, scalar dimensions and addresses: all 42 expected touch sensors confirmed.
- Actor and critic dimensions: 93 in each arm. Reset coordinates and masks checked over two reset seeds and multiple actual simulation steps. Nonzero synthetic raw readings confirm masking independently of whether incidental contact occurs at reset.
- PPO integration smoke: each arm completed one update with 2 environments and 2 rollout steps on CPU. Fresh evaluation runners loaded the resulting checkpoints with strict actor loading and produced finite actions.
- Both smoke arms began with identical actor and critic state hashes and parameter counts (28,804 actor; 28,673 critic).
- Initial actor hash in this CPU smoke: `5db1d43465074cf829b112ed6aa8a0de1566c5b8b9797196ce58ffb8ce94f495`.
- Initial critic hash in this CPU smoke: `b2723494b479c77ba0c795f424ce3a192c56051051306d48ced7f98846c3b169`.
- Full new evaluator path exercised for both arms with an external QA harness that shortened the horizon to 0.02 s: strict checkpoint load, actual actor/critic mask checks, terminal recording, episode CSV, summary JSON, trace and checksum generation passed. The delivered task retains its full 8 s horizon.
- Report pipeline exercised with explicitly synthetic fixtures: PNG/PDF rendering, paired-seed table and JSON/Markdown output passed. A single pair remained exploratory. Missing paired runs and modified evidence were rejected. Inference checks covered zero observed variance, small-sample sign-flip resolution, beneficial and inconclusive examples.
- Official training CLI discovers both installed task IDs and accepts the flags used by the launcher.
- Python compilation and `git diff --check`: passed.
- The XML/assets, original environment and MDP implementations, PPO defaults and dependency lockfile are unchanged from parent `8e985fffdf60cd6c0ba453f053bc1d3b6cd9e654`.

The QA runtime used Torch 2.13.0+cpu and Torchvision 0.28.0+cpu, with the remaining applicable pinned dependencies. There is no CUDA device in this environment. The real 256-environment, 500-update paired experiment and full-horizon GPU evaluation remain to be executed on NYK's workstation. The launcher first reruns the live preflight on cuda:0. No GPU throughput, real training success, or touch-effect conclusion is claimed here.
