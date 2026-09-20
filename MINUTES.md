# Microduck RL - Project Minutes & Running Summary

## Executive Summary
This document serves as an ongoing retrospective log and meeting minutes for the Microduck Reinforcement Learning (RL) stack setup, simulation training, and physical robot deployment project. It includes technical engineering milestones, workflow decisions, tooling choices, visual asset configurations, and collaboration strategies.

## Primary Project Goals
1. **End-to-End RL Pipeline:** Train, export, and evaluate locomotion policies across Cloud CUDA GPU and local macOS environments.
2. **Custom Reward Engineering:** Design, write, and tune mathematical reward/penalty functions (in `microduck_velocity_env_cfg.py` and `mdp.py`) to shape custom walking gaits and robot behaviors.
3. **Broad Robotics & Physical AI Mastery:** Build hands-on competency across RL simulation, PyTorch/PPO, ONNX execution, C++/Python environments, hardware teleoperation dynamics, and ROS2 autonomy.

---

## Technical Stack & Architecture

* **Local Workstation:** Apple Silicon M4 Max MacBook Pro (ARM64)
  * *Role:* Code editing, local evaluation (`play`), ONNX export verification, local VLM/VLA inference (Metal GPU via Ollama), and real-robot control.
* **Training Infrastructure:** Cloud CUDA GPUs (e.g., Google Colab T4/A100)
  * *Role:* Multi-environment PPO policy training via `mjlab` (requires NVIDIA CUDA).
* **Physical Robot:** Microduck
  * *OS:* Debian/Ubuntu Linux (ARM64) on Radxa Zero 3W.
  * *Control System:* Custom `robotd` background daemon managing a 50 Hz control loop over 15 servos.
* **Framework Stack:** `mjlab` (physics/RL framework based on MuJoCo Warp), Viser (default web viewer), PPO (RL algorithm), ONNX Runtime (deployment execution).

---

## Chronological Technical & Process Log

### 1. Environment Configuration & Setup (Completed)
* **Technical Milestones:**
  * Selected `uv` (Rust-based Python package manager) to manage environment dependencies and avoid Homebrew/pip `ensurepip` and `truststore` certificate bugs on macOS.
  * Python Version: Python 3.12 (`uv venv --python 3.12`).
  * Dependencies Installed: `mjlab-microduck`, `better-actuator-models` (BAM), `viser`, `torch`, `onnxruntime`, `warp-lang`.
* **Process & Meta Notes:**
  * Standardized on `uv` to maintain zero-friction reproducibility across Mac local and Cloud GPU environments.

### 2. Simulator & Framework Verification (Completed)
* **Technical Milestones:**
  * Verified interactive 3D simulation using Viser via `uv run play Mjlab-Velocity-Flat-MicroDuck --agent <zero|random>`.
  * Identified that `mjlab` uses `play` for interactive simulation playback rather than a standalone `viewer` module.
* **Process & Meta Notes:**
  * Established Viser as the path of least resistance for local 3D rendering and manual testing.

### 3. Hardware Architecture & Training Route Clarification (Completed)
* **Technical Milestones:**
  * Attempted local `train` execution on macOS; confirmed `mjlab` requires NVIDIA CUDA kernels for multi-environment parallel simulation.
  * Decoupled Workflow Established:
    1. *Train* PPO policies on free Cloud CUDA GPUs (Google Colab).
    2. *Download* checkpoint files (`.pt` / `.onnx`) to local Mac.
    3. *Evaluate* gait in local Viser viewer via `uv run play`.
* **Process & Meta Notes:**
  * Clarified role of local Apple Silicon (Metal/Ollama for high-level VLM/VLA spatial tasks in Phase 3) vs Cloud NVIDIA CUDA (low-level physics simulation in Phase 1-2).

### 4. Codebase Navigation & Configuration Mapping (Completed)
* **Technical Milestones:**
  * Mapped the conceptual "walk config" to its exact repository path: `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`.
  * Inspected `RewardsCfg` parameters, specifically analyzing exponential upright posture tuning (`weight=2.0`, `std^2=0.05`) designed to suppress steady forward pitch lean (~4°).
* **Process & Meta Notes:**
  * Identified discrepancy between conversational shorthand (`walk_config.py`) and exact repo file paths (`microduck_velocity_env_cfg.py`).
  * Unpacked reward engineering methodology: showing how empirical eval data (push-recovery fall distributions) drives mathematical weight adjustments.

### 5. Collaboration Strategy & Documentation Guidelines (Completed)
* **Process & Meta Notes:**
  * Decision made to track meta-conversations, process choices, and workflow rationale directly alongside technical milestones in a dual-track chronological log.
  * Established automated background logging strategy to eliminate manual prompting overhead.

### 6. Goal Alignment: Custom Reward Engineering (Completed)
* **Process & Meta Notes:**
  * Formally registered **Hands-on Reward Engineering** as a primary objective for the project.
  * Defined experimental A/B testing workflow: baseline training -> config modification -> comparative local evaluation (`play`).
  * Adopted Weights & Biases (WandB) for live experiment tracking, configured via private personal profile.

### 7. Baseline Policy Training & Convergence Dynamics (Completed)
* **Technical Milestones:**
  * Initiated and ran baseline PPO training in Google Colab T4 across 4,096 parallel environments.
  * Telemetry confirmed mean rewards climbed steadily, with episode lengths converging at the ~3.5s max timeout threshold (175–200 steps at 50 Hz).
  * Identified that the sinusoidal wave in survival time is expected behavior resulting from automatic 2–4 second command velocity resampling and PPO batch updates.
* **Process & Meta Notes:**
  * Validated that survival reaching ~3.5s confirms baseline dynamic balance and fall prevention.

### 8. Hardware Twin Visual Styling (`robot_walk.xml`) (Completed)
* **Technical Milestones:**
  * Customized MJCF material definitions in `robot_walk.xml` to match the physical Lavender edition hardware.
  * Set RGBA `0.72 0.58 0.84 1` (Lavender) across body shells, upper legs, and soles.
  * Set RGBA `0.55 0.80 0.88 1` (Teal) on `noenoeil_material` (eye bezel).
  * Set RGBA `0.96 0.82 0.15 1` (Bright Yellow) across beak, jaw, ankles, feet, and lower head shell.
* **Process & Meta Notes:**
  * Verified 1:1 visual parity between physical hardware and local Viser simulation rendering.

### 9. Session Resumption, Ephemeral Storage & CLI Troubleshooting (Completed)
* **Technical Milestones:**
  * Handled Colab disconnection and local drive wiping (`logs/rsl_rl/velocity` cleared).
  * Clarified local execution mechanics: Viser stays active on macOS after Colab disconnection because policy weights are downloaded and cached locally during `play`.
  * Resolved `tyro` CLI argument errors (`Unrecognized options: --resume_path`) by applying correct syntax: `--agent.resume True --agent.load-checkpoint model_750.pt`.
  * Corrected iteration math: set `--agent.max-iterations 1250` when resuming from step 750 to achieve the desired 2,000 total iteration target (~50–65 minutes execution on T4 GPU).
* **Process & Meta Notes:**
  * Established protocol to rename verified cached weights to `model_baseline.pt` on local drive to safeguard baseline benchmarks before starting A/B testing.

### 10. Firmware Baseline Calibration & Goal Refinement (Completed)
* **Technical Milestones:**
  * Calibrated benchmark expectations: recognized that Microduck ships with 7 factory pre-trained RL policies (Pollen Robotics / Hugging Face) using the same `microduck_rl` stack.
  * Defined target goals for custom policy Run #2: outperform factory safety limits by engineering higher top speed, improved turning agility, and optimized motor torque limits via `microduck_velocity_env_cfg.py`.

### 11. Tooling Rationale: `uv` vs `pixi` (Completed)
* **Process & Meta Notes:**
  * Evaluated `pixi` (Rust-based Conda/system manager common in robotics) vs `uv` (Rust-based PyPI package manager).
  * Standardized on `uv` because `mjlab`, PyTorch, and Viser are pure PyPI wheel distributions, ensuring lightweight virtualenv management across macOS and Google Colab.

### 12. Local Playback CLI Syntax Disambiguation (Completed)
* **Technical Milestones:**
  * Resolved CLI parameter mismatch when launching local evaluation via `uv run play`.
  * Identified that `play.py` uses `--checkpoint-file` for direct path loading or `--wandb-checkpoint-name` for WandB cache lookups rather than the generic `--checkpoint` flag.
  * Clarified local file path targets: `logs/rsl_rl/velocity/<RUN_ID>/model_3000.pt`.

### 13. Reward Engineering Strategy for Run #2 (Formulated)
* **Technical Milestones:**
  * Analyzed reward/penalty trade-offs to prevent forward-pitch collapse during high-velocity commands.
  * Established initial coupled weight edits for `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`:
    * `orient_l2`: `-1.0` -> `-2.5` (penalizes pitch/roll tilt).
    * `tracking_lin_vel`: `1.0` -> `1.5` (incentivizes forward speed).
    * `tracking_ang_vel`: `0.5` -> `0.85` (sharpens yaw turning response).
    * `action_rate_l2`: `-0.001` -> `-0.005` (suppresses servo chatter/jitter).
* **Process & Meta Notes:**
  * Defined rule of thumb for reward tuning: edit coupled hypothesis pairs (reward + stabilizing penalty) during exploratory phases, saving single-variable ablations for final optimization.

### 14. Interactive Viser Control UI Navigation (Completed)
* **Technical Milestones:**
  * Mapped Viser web UI right-side control panel layout:
    * `Enable` checkbox: Toggles live manual override mode.
    * `lin_vel_x` slider: Controls forward/backward target speed (v_x).
    * `lin_vel_y` slider: Controls lateral strafing speed (v_y).
    * `ang_vel_z` slider: Controls turning rate (omega_z).
  * Established standardized test target vectors for manual evaluation: 0.0 m/s (stance), 0.3 m/s (nominal walk), and 0.6 m/s (top speed).

### 15. Simulation Extension Options: Terrains & WASD Teleop (Documented)
* **Technical Milestones:**
  * Documented static MJCF geometry injection (`<geom type="box">` ramps and obstacles in `robot_walk.xml`) for quick visual testing vs procedural heightfield grids via `Mjlab-Velocity-Rough-MicroDuck`.
  * Outlined WASD teleoperation keyboard hook using `pynput` inside local Python evaluation scripts to dynamically update command velocity vectors.

### 16. Local Evaluation Telemetry & Step Counter Patch (Completed)
* **Technical Milestones:**
  * Identified that local Viser playback locks physics to 1x real-time speed while suppressing console stdout step logs.
  * Created programmatic evaluation patch for `.venv/lib/python3.12/site-packages/mjlab/scripts/play.py`:
    ```python
    _original_step = env.step
    _step_counter = 0

    def _tracking_step(action):
      nonlocal _step_counter
      _step_counter += 1
      obs, rew, dones, infos = _original_step(action)
      if dones.any():
        survival_s = _step_counter / 50.0
        print(f"\n[EVAL METRIC] Reset triggered at Step {_step_counter} | Survival Time: {survival_s:.2f}s")
        _step_counter = 0
      return obs, rew, dones, infos

    env.step = _tracking_step
    ```
* **Process & Meta Notes:**
  * Eliminated manual stopwatch timing in favor of deterministic Time = Steps / 50 Hz terminal logging.

### 17. Headless Benchmarking & Custom Evaluator Architecture (Designed)
* **Technical Milestones:**
  * Designed `benchmark_policy.py` for headless multi-trial evaluation (N=5 trials across v_x in [0.0, 0.3, 0.6] m/s) exporting CSV records containing survival duration, step counts, and velocity tracking RMSE.
  * Outlined modular evaluator CLI (`evaluator.py`) scope to sweep 3D command grids (v_x, v_y, omega_z), apply impulse pushes (5 N - 20 N), and measure Cost of Transport (CoT).

---

## Next Steps & Action Items

1. **Apply Run #2 Code Updates:** Save modified reward weights in `microduck_velocity_env_cfg.py` (`orient_l2=-2.5`, `tracking_lin_vel=1.5`, `action_rate_l2=-0.005`).
2. **Launch Retrain in Colab:** Upload updated configuration file to Google Colab and launch Run #2 training.
3. **Execute Comparative Evaluation:** Run local evaluation using `benchmark_policy.py` or patched `play.py` on `model_3000.pt` vs the new Run #2 checkpoint to verify stability gains.

---

## 2026-09-05 — Learning-plan and collaboration clarification

### Project direction

The project owner clarified that Microduck RL is intended to be a hands-on
learning project, not an agent-operated batch of experiments. The owner wants
to perform commands, edits, training, evaluation, and analysis personally in
order to build practical robotics engineering skill.

### Working agreement

Agents should:

1. explain the next action and why it matters;
2. wait for explicit permission before running commands or editing files;
3. let the owner perform the hands-on step;
4. review the resulting output together;
5. append a clear summary of the milestone to this file.

This journal records technical checkpoints as well as clarifications,
decisions, learning moments, project goals, and speculative ideas. It is
chronological and may contain ideas that are not yet part of the agreed plan.

### Stable plan location

Created `MICRODUCK_RL_LEARNING_PLAN.md` in the repository root. That file is
the stable learner-facing roadmap: goals, milestones, learning sequence,
evaluation methodology, safety rules, and future-skill backlog. This file
remains the chronological project journal; the two documents serve different
purposes and should not duplicate each other.

### Technical checkpoint

Before this clarification, an agent attempted the five-iteration CUDA smoke
test on the Mac:

```text
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 64 \
    --env.seed 123 \
    --agent.max-iterations 5
```

Training did not start. The installed trainer reached GPU selection and failed
with `IndexError: list index out of range` because the Mac has no CUDA GPU.
No baseline training run was produced. Future training should be performed by
the owner on an approved Linux/NVIDIA or managed-cloud machine.

### Repository hygiene

An evaluator scaffold was briefly created during the mistaken autonomous
execution and was then deleted at the owner's request. No evaluator is
currently part of the repository. The pre-existing `robot_walk.xml`
modification, `.DS_Store`, and `MINUTES.md` working-tree state were not
overwritten or cleaned up.

---

## 2026-09-05 — Phase roadmap expanded

### Current Phase 1 sequence

The owner clarified the immediate project sequence:

1. environment setup — done-ish;
2. first walking policy — already trained;
3. change selected reward weights;
4. train a second policy;
5. perform A/B evaluation with a reusable walking evaluator;
6. implement and study an original reward function.

The stable roadmap now divides this into Phase 1A–1F. “Done” is treated as
“performed once”; provenance, reproducibility, evaluation, and understanding
still need to be completed before a result is considered an engineering
baseline.

### Longer-term direction

Phase 2 is intentionally divided into branches rather than a single
unbounded list:

- robotics software foundations: Linux, C++, testing, Docker, profiling;
- controls and robot systems: kinematics, dynamics, timing, latency, safety;
- ROS2 and navigation: nodes, transforms, bags, SLAM, planning, recovery;
- perception and spatial intelligence: depth, point clouds, calibration,
  tracking, scene representations;
- VLM/VLA and embodied AI: multimodal interfaces, data, latency, safety;
- an end-to-end application demo built from individually validated skills.

The decision rule is to pursue depth in the current phase, then branch when a
phase gate is met. Interesting ideas can be promoted through a short decision
card containing the idea, purpose, smallest experiment, resources, metric,
time limit, and kill condition.

### Humanoid career alignment

Public Humanoid role descriptions were reviewed as a direction-setting
reference. The recurring engineering areas were reinforcement-learning
locomotion/loco-manipulation, high-fidelity simulation, model-based and
learning-based control, real-time production software, ROS2/C++/Python,
perception/spatial intelligence, and deployment of ML systems on robot
hardware.

The project goal is not to imitate the entire Humanoid stack on Microduck.
It is to use the duck as a small laboratory for transferable engineering
habits: explicit interfaces, reproducible experiments, measured behavior,
failure analysis, testing, and safe deployment.

### Repository change

Expanded `MICRODUCK_RL_LEARNING_PLAN.md` with:

- current Phase 1A–1F subdivisions;
- a Humanoid-aligned career section;
- Phase 2A–2F learning branches;
- a decision process for new or divergent ideas;
- a staged end-to-end demo concept.

No training, evaluation, cloud submission, policy-code change, or hardware
action was performed during this update.

---

## 2026-09-05 — Phase 1B: Policy A provenance

Phase 1B did not train a new walk. The first Colab walking run already
existed; this session reconstructed which W&B run, checkpoint, seed, and git
commit that policy actually is, so later A/B work has a named baseline
(**Policy A**) instead of “whatever `.pt` is in the cache.”

### Policy A card

Use this object, and only this object, as the Phase 1 walking baseline.

| Field | Value |
|---|---|
| Label | Policy A (flat walk baseline) |
| Task | `Mjlab-Velocity-Flat-MicroDuck` |
| W&B run | `danielmason2206-personal/mjlab_microduck/mdnezvfu` |
| URL | https://wandb.ai/danielmason2206-personal/mjlab_microduck/runs/mdnezvfu |
| Checkpoint | `model_3000.pt` (PPO iteration 3000; 24 env steps per iteration) |
| Play / export | `--wandb-run-path danielmason2206-personal/mjlab_microduck/mdnezvfu` with checkpoint 3000 |
| Seed | 42 |
| Parallel envs | 4096 |
| GPU | 1× Tesla T4 (Colab) |
| Train git | `29e887ecfbf5d37144759e5a9f8a176dfb83d547` (W&B recorded clean) |
| Local git at documentation | same commit; **dirty** `src/mjlab_microduck/robot/microduck/robot_walk.xml` (lavender appearance only — not in the trained weights; do not treat it as part of a reward A/B) |
| Obs normalizer | on (`train_cfg.actor.obs_normalization: true`) — ONNX must go through `scripts/export.py` |
| Resume | `--agent.resume True --agent.load-checkpoint model_1250.pt`; `max_iterations` was 10000; the run **crashed** after uploading through `model_3000.pt` |
| Last W&B train stats | `Train/mean_reward` ≈ 113.61; `Train/mean_episode_length` ≈ 960.26 |

### Resume chain (same git, three W&B runs)

Colab disconnects produced four project runs. Ignore `1anjypmd` (killed at
`model_0.pt`). The surviving walk is one training story split across:

1. `391ofvzp` — iters 0→750 (`model_0.pt` … `model_750.pt`)
2. `r5t0wn2e` — 750→1250
3. `mdnezvfu` — 1250→3000 ← **Policy A lives here**

### Local cache footgun

A file exists at:

`logs/rsl_rl/velocity/wandb_checkpoints/391ofvzp/model_3000.pt`

Byte size `4843445` matches W&B `mdnezvfu`’s `model_3000.pt`. The **folder
name is wrong**: run `391ofvzp` never uploaded a 3000-iter checkpoint. Do not
identify Policy A as `391ofvzp`. Always name the W&B run `mdnezvfu`.

That cache directory also contains `model_250.pt` and `model_1250.pt` from
earlier segments.

### Reward-term snapshot (weighted `Episode_Reward/*` at end of `mdnezvfu`)

Penalties were ≤ 0 (sign-convention check passed). `body_pose_tracking = 0`
is the unused command slot.

- `action_rate_l2` −1.038
- `air_time` +1.015
- `angular_momentum` ≈ 0
- `body_ang_vel` −0.030
- `body_pose_tracking` 0
- `dof_pos_limits` −0.004
- `foot_clearance` −0.005
- `foot_slip` ≈ 0
- `foot_swing_height` −0.003
- `head_pose_bias` −0.315
- `head_pose_tracking` +1.658
- `pose` +0.609
- `self_collisions` −0.001
- `track_angular_velocity` +0.722
- `track_linear_velocity` +1.308
- `upright` +1.738

This snapshot is the training log at crash time, not a walking benchmark.

### Status

Phase 1B identification is recorded. No playback, ONNX export, reward-config
edit, or Policy B training was done in this session. Phase 1C (one controlled
reward change) has not started.

---

## 2026-09-20 — Phase 1C–1E: 2× smoothness, walking eval, idle basin

Work from this laptop plus Colab, spanning git hygiene, Policy B training,
and the first measured A/B. Lavender livery was kept **out** of the reward
comparison (own commit on `main`; XML left stock).

### Git

- Fork: https://github.com/DanMason6022/microduck_rl (`origin` / `mine`).
- `upstream` = `pollen-robotics/microduck_rl` (fetch only; do not push).
- Experiment branch: `phase-1c-changing-training-weights` (tracks the fork).
- Fork `main` tip at documentation time: livery commit `c470887`.

### Phase 1C / 1D — Policy B

Hypothesis: 2× the whole `action_rate_l2` curriculum vs Policy A, nothing else
changed (not the old four-weight “Run #2” note).

A (OOB) stages: `-0.1, -0.2, -0.4, -0.6, -0.8, -1.0` by iter 1500.  
B (trained): each stage ×2, ending **-2.0**.

| Field | Policy B |
|---|---|
| Task | `Mjlab-Velocity-Flat-MicroDuck` |
| W&B | `danielmason2206-personal/mjlab_microduck/arbfiaxm` |
| Checkpoint used in eval | `model_2999.pt` (local cache; 1 iter off 3000) |
| Seed / envs / budget | 42 / 4096 / ~3000 iters (matched to A) |
| GPU | Colab T4 |

W&B at ~3k looked similar to A except `Episode_Reward/track_angular_velocity`
~0.74 → ~0.81. That is a **weighted training term**, not “turns better.”

### Phase 1E — walking battery

- Built headless [`scripts/eval_walk.py`](scripts/eval_walk.py) (mjlab `.pt` /
  same stack as `play`): frozen twist, no pushes, CSV/JSON.
- Do **not** use [`scripts/eval_and_promote.py`](scripts/eval_and_promote.py)
  (calls missing `benchmark_policy`, dummy `0.050` on parse fail).
- Protocol: 10 s, seeds 0–4, stand / vx 0.2 / 0.4 / ±vy / ±wz.
- Outputs: `results/eval_A/`, `results/eval_B/`,
  `results/eval_AB/compare.json` and `compare.txt`.
- Eval A checkpoint path was the misnamed cache folder `391ofvzp`; bytes match
  Policy A (`mdnezvfu`). Still name A as `mdnezvfu`.

**Result (35/35 trials, no falls on either policy):** zero falls is expected
(flat, no pushes, 10 s). B did **not** walk better.

| Case | A | B | Read |
|---|---|---|---|
| stand action-rate RMS | 0.079 | 0.061 | B slightly quieter (real) |
| vx=0.2 lin RMSE | 0.175 | **0.199** | B error ≈ the command |
| vx=0.2 xy_drift | 0.78 m | **1.99 m** | missing the 2 m walk |
| turn ±0.5 ang RMSE | 0.23–0.31 | **~0.49** | B not spinning |

B’s action-rate collapse on `vx=0.2` matches **stand**, not a calm gait.
Tracking Gaussian `std ≈ 0.32` m/s still pays ~67% if you stand vs a 0.2 m/s
command, so a heavy action-rate tax from iter 0 makes idle the cheap basin.

Automatic compare label `B_wins_smoothness` is **wrong for walking**: overall
lin RMSE helped B because stand/turn have `vx=0`. **A remains champion.**

Operator check (no agent): Viser at vx=0.2 — did it leave the spawn? CSV:
`lin_vel_rmse` glued to 0.2 and `xy_drift` ≈ 2 m ⇒ freeze.

### Policy C (queued)

Not trained. Colab quota exhausted. Stay on Colab; HF Jobs is the preferred
repo path later (`l4x1` / T4-medium, smoke first). Between resets: Mac eval,
ONNX rehearsal, 1F spec on paper — not another 2× schedule.

C config intent: **A’s stages through 1250**, only iter **1500 = −1.25**.
Same 3000 iters, seed 42, 4096 envs. Must pass `--agent.max-iterations 3000`.
A wandb/HF run name is a **label** (notes or `--agent.run-name`), not a file.

Pass for C vs A: walks at 0.2 (drift not ~2 m) **and** quieter on those
walking trials. Freeze again ⇒ A stays champion.

### Status

1A–1E first loop complete. 1F not started. C not trained. Working tree on
`phase-1c-changing-training-weights` also holds uncommitted eval code, results,
and the C `action_rate` stages — not yet a clean push.