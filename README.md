# learned_rl_study

A small benchmark of two released *learned* reinforcement-learning components against hand-written
baselines, on nine small JAX environments:

- **Disco103**, the meta-learned update rule released with
  [DiscoRL](https://github.com/google-deepmind/disco_rl) (Oh et al., Nature 2025). It replaces the
  loss function: it tells the agent what its policy and prediction outputs should move towards.
- **OPEN**, the learned optimizer released with
  [rl-learned-optimization](https://github.com/AlexGoldie/rl-learned-optimization) (Goldie et al.,
  NeurIPS 2024). It replaces Adam inside PPO: given a gradient, it returns the parameter update.

Both are used with their released weights, frozen. Only the agent is trained.

**Results:** [REPORT.md](REPORT.md) has the tables and figures, `results/` the data behind them, and
`results/dashboard.html` the learning curves as interactive charts (download it and open it in a
browser).

## Methods

| Method | What is trained how | `--method` of `train_jax.py` |
|---|---|---|
| REINFORCE | Vanilla policy gradient: complete episodes, Monte Carlo reward-to-go, no baseline, no critic, no entropy bonus | `reinforce` |
| A2C | Synchronous advantage actor-critic, 8-step rollouts, GAE | `a2c` |
| PPO | Clipped-objective PPO, 4 epochs over each batch | `ppo` |
| PPO + OPEN | PPO in the training loop OPEN was meta-trained in, with OPEN as the optimizer | `open` |
| PPO + Adam, OPEN's loop | The same loop with Adam, as the matched reference for OPEN | `pporef` |
| Disco103 | A `disco_rl` agent trained by the Disco103 rule from a replay buffer | `disco` |

Variants that are run as separate methods:

- Disco103 with a 64×64 agent torso (`--torso 64 64 --model_size 64`) and with the 512×512 torso.
- Disco103 with replay ratio 32, as in the DiscoRL example notebook, and with replay ratio 4.
- REINFORCE, A2C and PPO with 512×512 networks in addition to 64×64.

`train_jax.py` trains every method on [gymnax](https://github.com/RobertTLange/gymnax) environments
with the environment, the agent and the update compiled into one JAX program, and all seeds of a
setting in one process. `train_reinforce.py`, `train_a2c.py`, `train_ppo.py`, `train_open.py` and
`train_disco.py` are single-seed versions of the same methods for Gymnasium environments;
`train_jax.py` takes its networks and update code from them. The results in this repository are
all from `train_jax.py`.

## Environments

| Name in `bench.py` | gymnax environment | Steps per run |
|---|---|---|
| `catch` | `Catch-bsuite` | 200,000 |
| `cartpole` | `CartPole-v1` | 500,000 |
| `acrobot` | `Acrobot-v1` | 500,000 |
| `fourrooms` | `FourRooms-misc` | 300,000 |
| `deepsea` | `DeepSea-bsuite` (size 8) | 300,000 |
| `breakout` | `Breakout-MinAtar` | 1,500,000 |
| `spaceinvaders` | `SpaceInvaders-MinAtar` | 1,500,000 |
| `asterix` | `Asterix-MinAtar` | 1,500,000 |
| `freeway` | `Freeway-MinAtar` | 1,500,000 |

All methods use 16 environments stepped together and each environment's own episode limit.
Observations are flattened. `bench.py` also lists two LunarLander configurations for the Gymnasium
scripts; they are not part of the results.

## Protocol

`bench.py` runs the same procedure for every environment.

1. **Search.** Each method has one searched setting: the learning rate, or for OPEN which released
   weights to use (`gridworld` or `multi`; on a MinAtar game also the weights meta-trained on that
   game alone). Every value is run with seeds 0, 1 and 2. A value's
   score is the final return, meaning the mean return of the last 100 episodes of a run, averaged
   over the three seeds. Exact ties are broken by the mean return over all episodes.
2. **Extension.** If the best learning rate is the lowest or highest value tried, the next value in
   that direction is run as well, at most twice per method.
3. **Evaluation.** The best value is run with ten new seeds, 10 to 19. These runs are the results.

The starting values of every search are listed in `bench.py`. All other hyperparameters are fixed
and are the defaults of the training scripts.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or 3.12.

```sh
uv sync
```

On a Linux machine with an NVIDIA GPU, install JAX with CUDA support instead:

```sh
uv sync --extra cuda
```

Training then runs on the GPU, and several processes share it; `bench.py ... --parallel N` sets how
many run at once. The results in this repository were made this way with `--parallel 3`.

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

# One setting by hand: three seeds in one process, run folders runs/example_s0, _s1, _s2.
uv run python train_jax.py --method ppo --out runs/example --env CartPole-v1 --env_steps 500000 --lr 1e-3 --seeds 0 1 2

# REPORT.md, results/ and results/dashboard.html from the runs under runs/.
uv run python report.py

# Sanity check: train an agent on Catch with the Disco103 rule (Gymnasium-style loop).
uv run python catch.py --rule disco --seed 0
```

`bench.py` runs 8 training processes at a time (`--parallel`), each limited to one CPU thread, and
skips runs that already finished, so it can be interrupted and resumed. Disco103 with replay ratio
32 is the slowest method: on an RTX 4060 Ti it takes about 1.2 to 1.6 GPU-minutes per seed per
million environment steps, and the whole benchmark took about 16 hours.

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

The limitations of the setup are listed with the results, in [REPORT.md](REPORT.md#limitations). In
short: the methods are matched on environment steps only; one setting per method is searched, with
three seeds; the budgets are small; several environments separate the methods little; and OPEN is
not run in its own configuration.

## Tests

```sh
uv run python tests/test_open.py
uv run python tests/test_disco_unread.py
```

`open_optimizer.py` is a re-implementation of OPEN without the original's dependencies. The test
feeds recorded inputs to it and compares the parameters after several consecutive optimizer steps
with the ones the original code produced (`tests/open_fixture.npz`, made by
`tests/make_open_fixture.py`).

`tests/test_disco_unread.py` checks that the Disco103 learner step gives exactly the same result
when the agent outputs `q`, `y`, `z` and `aux_pi` stored with a rollout are replaced by zeros. The
replay buffer of `train_jax.py` relies on this and does not store them.

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
- Environments: [gymnax](https://github.com/RobertTLange/gymnax) (Lange, 2022), which includes JAX
  versions of bsuite and MinAtar tasks; the single-seed scripts use
  [Gymnasium](https://gymnasium.farama.org/) and [MinAtar](https://github.com/kenjyoung/MinAtar).
