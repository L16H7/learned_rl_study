# learned_rl_study

A small benchmark of two released *learned* reinforcement-learning components against hand-written
baselines, on tasks that train on a laptop CPU:

- **Disco103**, the meta-learned update rule released with
  [DiscoRL](https://github.com/google-deepmind/disco_rl) (Oh et al., Nature 2025). It replaces the
  loss function: it tells the agent what its policy and prediction outputs should move towards.
- **OPEN**, the learned optimizer released with
  [rl-learned-optimization](https://github.com/AlexGoldie/rl-learned-optimization) (Goldie et al.,
  NeurIPS 2024). It replaces Adam inside PPO: given a gradient, it returns the parameter update.

Both are used with their released weights, frozen. Only the agent is trained.

**Status: work in progress.** The benchmark runs are being produced. Results, the run data and a
report will be added to this repository when they are complete. Nothing here should be read as a
result yet.

## Methods

| Method | What is trained how | Script |
|---|---|---|
| REINFORCE | Vanilla policy gradient: complete episodes, Monte Carlo reward-to-go, no baseline, no critic, no entropy bonus | `train_reinforce.py` |
| A2C | Synchronous advantage actor-critic, 8-step rollouts, GAE | `train_a2c.py` |
| PPO | Clipped-objective PPO, 4 epochs over each batch | `train_ppo.py` |
| PPO + OPEN | PPO in the training loop OPEN was meta-trained in, with OPEN as the optimizer | `train_open.py` |
| PPO + Adam, OPEN's loop | The same loop with Adam, as the matched reference for OPEN | `train_open.py --optimizer adam` |
| Disco103 | A `disco_rl` agent trained by the Disco103 rule from a replay buffer | `train_disco.py` |

Variants that are run as separate methods:

- Disco103 with a 64×64 agent torso (`--torso 64 64 --model_size 64`) and with the 512×512 torso.
- Disco103 with replay ratio 32, as in the DiscoRL example notebook, and with replay ratio 4.
- REINFORCE, A2C and PPO with 512×512 networks in addition to 64×64.

## Environments

| Name in `bench.py` | Environment | Steps per run | Episode limit |
|---|---|---|---|
| `cartpole` | `CartPole-v1` | 500,000 | the environment's own (500) |
| `acrobot` | `Acrobot-v1` | 500,000 | the environment's own (500) |
| `lunarlander` | `LunarLander-v3` | 1,000,000 | 400 steps |
| `lunarlander_penalty` | `LunarLander-v3`, training reward reduced by 100 when the limit is hit | 1,000,000 | 400 steps |
| `breakout` | `MinAtar/Breakout-v1`, observation flattened | 1,000,000 | 1,000 steps |

All methods use 16 environments stepped together. Reported returns always use the environment's
original reward.

## Protocol

`bench.py` runs the same procedure for every environment.

1. **Search.** Each method has one searched setting: the learning rate, or for OPEN which released
   weights to use (`gridworld` or `multi`; on MinAtar Breakout also `breakout`, the weights
   meta-trained on that game alone). Every value is run with seeds 0 and 1. A value's score
   is the final return, meaning the mean return of the last 100 episodes of a run, averaged over
   the two seeds. Exact ties are broken by the mean return over all episodes.
2. **Extension.** If the best learning rate is the lowest or highest value tried, the next value in
   that direction is run as well, at most twice per method.
3. **Evaluation.** The best value is run with five new seeds, 10 to 14. These runs are the results.

The starting values of every search are listed in `bench.py`. All other hyperparameters are fixed
and are the defaults of the training scripts.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or 3.12.

```sh
uv sync
```

The Disco103 weights (2.8 MB) and the OPEN weights are downloaded from their repositories on first
use and stored in `weights/`. They are not part of this repository. JAX is pinned to 0.6.2 because
`disco_rl` calls functions that newer JAX releases removed.

## Run

```sh
# The whole benchmark for one environment: search, then evaluation.
uv run python bench.py cartpole all

# Only some methods, or only one phase.
uv run python bench.py cartpole all --groups main
uv run python bench.py cartpole tune --methods ppo a2c
uv run python bench.py cartpole final

# One training run by hand.
uv run python train_ppo.py --out runs/example --env CartPole-v1 --env_steps 500000 --lr 1e-3 --seed 0

# Learning curves and a summary table for a folder of runs.
uv run python plot.py runs/cartpole_final --one_figure

# Score a trained agent and save a GIF of it (environments that can render).
uv run python watch.py runs/cartpole_final/ppo_lr1e-3_s10

# Sanity check: train an agent on Catch with the Disco103 rule.
uv run python catch.py --rule disco --seed 0
```

`bench.py` runs 8 training processes at a time, each limited to one thread, and skips runs that
already finished, so it can be interrupted and resumed. Disco103 with replay ratio 32 is by far the
slowest method: one 500,000-step run takes about an hour or more on an Apple M3 Pro.

Each run folder under `runs/` holds:

- `args.json`: every argument of the run.
- `progress.csv`: one row every 10,000 environment steps.
- `episodes.csv`: return and length of every episode.
- `params.pkl`: the latest agent parameters.
- `provenance.json`: the command, a hash of the source files and the git commit (written by `bench.py`).

## Training setup of each method

| | REINFORCE | A2C | PPO | PPO + OPEN / PPO + Adam, OPEN's loop | Disco103 |
|---|---|---|---|---|---|
| Data per update | one complete episode from each of the 16 environments | 16 × 8 steps | 16 × 64 steps, split into 8 minibatches, 4 epochs | same as PPO | 64 trajectories of 29 steps sampled from a buffer of 1,024 |
| Updates per 1,000 environment steps | depends on episode length | about 8 | about 31 | about 31 | about 17 at replay ratio 32, about 2 at ratio 4 |
| Value estimates | none | GAE (λ = 0.95) | GAE (λ = 0.95) | GAE (λ = 0.95) | the rule's own predictions |
| Networks | policy MLP | separate actor and critic MLPs | separate actor and critic MLPs | separate actor and critic MLPs | MLP torso, plus the heads and the action-conditional model the rule requires |
| Optimizer | Adam | Adam | Adam | OPEN, or Adam | `disco_rl` default |
| Gradient clipping | none | one global norm, 0.5 | one global norm, 0.5 | actor and critic separately, 0.5 | `disco_rl` default |
| Value loss | none | squared error | squared error | clipped, as in OPEN's training | set by the rule |

The discount is 0.99 for the baselines. Baseline networks use tanh activations and orthogonal
initialisation.

## Known limitations

- The methods are matched on environment steps and number of environments. They are not matched on
  the number of gradient updates, on network size (Disco103 needs prediction heads of fixed width
  600 and 601 per action, whatever the torso size) or on compute time.
- Only one setting per method is searched. Disco103's other settings are the `disco_rl` defaults,
  which were chosen for other environments.
- The search uses two seeds, so the choice of setting is noisy. The evaluation uses five seeds, so
  small differences between methods cannot be resolved.
- The budgets are small compared with the ones the learned components were evaluated with by their
  authors. On MinAtar Breakout, 1,000,000 steps is early in training for every method here.
- Reaching the step limit is treated as the end of the episode by every method: no value is
  bootstrapped from the state after the limit.
- OPEN is used outside the environments it was meta-trained on, except on MinAtar Breakout. Even
  there the setup differs from OPEN's own: a different implementation of the game, with the
  observation flattened.

## Tests

```sh
uv run python tests/test_open.py
```

`open_optimizer.py` is a re-implementation of OPEN without the original's dependencies. The test
feeds recorded inputs to it and compares the parameters after several consecutive optimizer steps
with the ones the original code produced (`tests/open_fixture.npz`, made by
`tests/make_open_fixture.py`).

## Credits

- DiscoRL code and the Disco103 weights: Oh et al., *Discovering state-of-the-art reinforcement
  learning algorithms*, Nature 2025, <https://github.com/google-deepmind/disco_rl>. Per the
  upstream README, the software is licensed under Apache 2.0 and other materials under CC-BY 4.0.
  `catch.py` and the training loop in `train_disco.py` are adapted from its evaluation notebook
  (`colabs/eval.ipynb`).
- OPEN and its weights: Goldie, Lu, Jackson, Whiteson, Foerster, *Can Learned Optimization Make
  Reinforcement Learning Less Difficult?*, NeurIPS 2024,
  <https://github.com/AlexGoldie/rl-learned-optimization>, Apache 2.0. `open_optimizer.py` is a
  re-implementation of `rl_optimizer/network.py`, and the loop in `train_open.py` follows
  `rl_optimizer/train.py`, both from that repository.
- A copy of the Apache 2.0 licence that covers the adapted upstream code is in
  `LICENSES/Apache-2.0.txt`.
- Environments: [Gymnasium](https://gymnasium.farama.org/) and
  [MinAtar](https://github.com/kenjyoung/MinAtar).
