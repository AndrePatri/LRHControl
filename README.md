![AugMPC logo](docs/images/logo_new.svg)

Reinforcement Learning-Augmented Model Predictive Control *at scale* for legged and hybrid robots. Part of the [IBRIDO](https://github.com/AndrePatri/IBRIDO) project.

## Architecture

<p align="center">
  <img src="docs/images/overview_compressed.png" alt="approach overview" width="700">
</p>

- **Hierarchical RL-MPC coupling** – The RL agent chooses contact schedules and twist commands for the underlying MPC controllers. A new flight phase is injected, for each limb, when the corresponding actions *instantaneously* exceed a given thresholds.

- **Sample efficient** – high-throughput experience generation thanks to careful MPC parallelization and vectorized simulation. 50+ rt factor, 800 envs/MPCs 20Hz, 1 s horizon,  Sample-efficient learning with MPCs in the loop with the Soft Actor Critic (SAC) algorithm, convergence from 1 up to 10x10^6 environment steps (6h wall time, 9-29 sim. days) VS > 100x10^6 (N.A., ~20 sim. days) of a typical proprioceptive full RL policy. 
<p align="center">
  <img src="docs/images/sub_rewards.png" alt="rewards" width="500">
</p>

- **Domain adaptability** – thanks to MPC's robustness, successful sim-to-sim and sim-to-real zero-shot transfer *without any domain randomization* (no contact properties, inertial, timing randomizations).

- **Robot adaptability** – validated on robots with *different morphologies and weight distributions* (30-120 Kg), with standard legged and *hybrid locomotion* tasks.

<p align="center">
  <img src="docs/images/intro_image_compressed.png" alt="sim-to-sim, sim-to-real" width="500">
</p>

- **Non-Gaited** contact scheduling – our architecture is able to generate completely acyclic gaits and timing adaptations:

<p align="center">
  <img src="docs/images/acyclic_sequence.png" alt="sim-to-sim, sim-to-real" width="700">
</p>
<p align="center">
  <img src="docs/hybrid_quadruped_fake_pos_track.gif" alt="sim-to-sim, sim-to-real" width="300">
</p>

## Software
<p align="center">
  <img src="docs/images/ibrido_arch_compressed.png" alt=" software overview" width="600">
</p>

**Shared-memory first design**: AugMPC relies on a shared memory layer, built on top of [EigenIPC](https://github.com/AndrePatri/EigenIPC) for deterministic, real-time-safe communication between simulators, controllers, and learning processes.

AugMPC’s is essentially made of three main components:

1. **World interface** – Implements `AugMPCWorldInterfaceBase`. It connects to Isaac Sim, xbot2, or hardware, publishes robot states, and triggers MPCHive controllers via shared memory. Optional remote stepping lets the training loop decide when the simulator should advance.
2. **MPC cluster** – Uses [MPCHive](https://github.com/AndrePatri/MPCViz)'s `ControlClusterServer/Client` to spawn multiple receding-horizon controllers (see `aug_mpc.controllers`). Each controller reads robot states, solves its MPC problem and writes predictions and commands back to shared memory.
3. **Training environment + RL algorithm** – An `AugMPCTrainingEnvBase` derivative defines the MDP at hand (observations, actions, rewards, terminations, trucations), which is then used by the training executable (SAC is the default, PPO supported).

Specific implentations of world interfaces and training environments are available at [AugMPCEnvs](https://github.com/AndrePatri/AugMPCEnvs).

## Repository layout

```
aug_mpc/
├── training_envs/     # Base classes and wrappers for AugMPCEnvs environments
├── world_interfaces/  # Base world interface that AugMPCEnvs extends
├── controllers/       # LRHC/MPCHive clients and Horizon-based MPC implementations
├── agents/            # Neural network policies (PPO, SAC, dummy)
├── training_algs/     # PPO/SAC trainers, rollout logic, persistence
├── scripts/           # Launchers for clusters, world interfaces, and training loops
└── utils/             # Shared-memory helpers, visualization bridges, teleop, math/utils
```

## Installation

The preferred way to install MPCHive is through [ibrido-containers](https://github.com/AndrePatri/ibrido-containers), which ships with all necessary dependencies.

## Extending AugMPC

1. **New controller** – 
2. **New world interface** – 
3. **New training environment** – 
4. **New agent/algorithm** – 