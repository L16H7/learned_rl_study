"""Shared pieces: the vector environment, networks, GAE, run logging and Disco103 weights."""
import json
import pickle
import time
import urllib.request
from pathlib import Path

import chex
import gymnasium as gym
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_ENV = "LunarLander-v3"


def make_env(env_id: str, max_episode_steps=None, **kwargs):
    """A Gymnasium environment; max_episode_steps=None keeps the environment's own step limit.
    MinAtar games ("MinAtar/Breakout-v1", ...) are registered on first use; they have no step limit
    of their own, so give one."""
    if env_id.startswith("MinAtar/") and env_id not in gym.registry:
        import minatar.gym

        minatar.gym.register_envs()
    return gym.make(env_id, max_episode_steps=max_episode_steps, **kwargs)


def env_dims(env_id: str):
    """(flattened observation size, number of discrete actions)."""
    env = make_env(env_id)
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError(f"{env_id} does not have discrete actions")
    return int(np.prod(env.observation_space.shape)), int(env.action_space.n)


class VecEnv:
    """`num_envs` independent copies of a Gymnasium environment with automatic reset.

    step() returns the observation after the action (the first observation of the next episode when
    the episode ended), the training reward, and whether the episode ended. Reaching a terminal state
    and reaching the step limit both count as the end of the episode. `finished` lists every completed
    episode as (return, length, total environment steps so far); returns use the original reward.
    """

    def __init__(self, env_id: str, num_envs: int, seed: int, max_episode_steps=None, timeout_penalty: float = 0.0):
        self.num_envs = num_envs
        self.envs = [make_env(env_id, max_episode_steps) for _ in range(num_envs)]
        self.obs_dim, self.num_actions = env_dims(env_id)
        self._seed = seed
        self._timeout_penalty = timeout_penalty  # subtracted from the training reward at the step limit
        self._returns = np.zeros(num_envs)
        self._lengths = np.zeros(num_envs, dtype=int)
        self.finished = []
        self.total_steps = 0

    def reset(self) -> np.ndarray:
        self._returns[:] = 0
        self._lengths[:] = 0
        obs = [env.reset(seed=self._seed * 1000 + i)[0] for i, env in enumerate(self.envs)]
        return np.stack(obs).reshape(self.num_envs, -1).astype(np.float32)

    def step(self, actions):
        obs = np.empty((self.num_envs, self.obs_dim), dtype=np.float32)
        rewards = np.empty(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        for i, (env, action) in enumerate(zip(self.envs, np.asarray(actions).reshape(-1))):
            o, r, terminated, truncated, _ = env.step(int(action))
            self.total_steps += 1
            self._returns[i] += r
            self._lengths[i] += 1
            if truncated and not terminated:
                r -= self._timeout_penalty
            if terminated or truncated:
                self.finished.append((float(self._returns[i]), int(self._lengths[i]), self.total_steps))
                self._returns[i], self._lengths[i] = 0.0, 0
                o, _ = env.reset()
                dones[i] = True
            obs[i], rewards[i] = np.asarray(o).reshape(-1), r
        return obs, rewards, dones


class RunLog:
    """Writes args.json once, then progress.csv, episodes.csv and params.pkl every `every` steps."""

    def __init__(self, out, args: dict, every: int = 10_000):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "args.json").write_text(json.dumps(args, indent=1))
        self._every, self._next, self._t0 = every, every, time.time()
        self._progress = open(self.out / "progress.csv", "w")
        self._progress.write("env_steps,updates,episodes,mean_return_last100,mean_length_last100,elapsed_s\n")

    def __call__(self, env_steps: int, updates: int, finished: list, params, final: bool = False):
        if env_steps < self._next and not final:
            return
        self._next = env_steps + self._every
        recent = finished[-100:]
        ret = np.mean([e[0] for e in recent]) if recent else float("nan")
        length = np.mean([e[1] for e in recent]) if recent else float("nan")
        line = f"{env_steps},{updates},{len(finished)},{ret:.1f},{length:.0f},{time.time() - self._t0:.0f}"
        self._progress.write(line + "\n")
        self._progress.flush()
        print(line, flush=True)
        with open(self.out / "params.pkl", "wb") as f:
            pickle.dump(jax.device_get(params), f)
        with open(self.out / "episodes.csv", "w") as f:
            f.write("episode,return,length,env_steps\n")
            f.writelines(f"{i},{r:.2f},{n},{s}\n" for i, (r, n, s) in enumerate(finished))


def _mlp(hidden, n_out: int, head_scale: float):
    def forward(x):
        for h in hidden:
            x = jax.nn.tanh(hk.Linear(h, w_init=hk.initializers.Orthogonal(np.sqrt(2)))(x))
        return hk.Linear(n_out, w_init=hk.initializers.Orthogonal(head_scale))(x)

    return hk.without_apply_rng(hk.transform(forward))


def policy_net(num_actions: int, hidden=(64, 64)):
    """Observation -> action logits. Tanh MLP with orthogonal initialisation."""
    return _mlp(tuple(hidden), num_actions, 0.01)


def value_net(hidden=(64, 64)):
    """Observation -> state value, shape [..., 1]."""
    return _mlp(tuple(hidden), 1, 1.0)


def gae(values, last_value, rewards, dones, gamma: float, lam: float):
    """Generalised advantage estimation.

    values, rewards, dones: [T, N]; last_value: [N], the value of the observation after the last step.
    dones[t] = 1 when the episode ended with step t; nothing is bootstrapped across that boundary.
    Returns (advantages, value targets), both [T, N].
    """
    next_values = jnp.concatenate([values[1:], last_value[None]], axis=0)
    not_done = 1.0 - dones
    deltas = rewards + gamma * next_values * not_done - values

    def step(carry, x):
        delta, nd = x
        adv = delta + gamma * lam * nd * carry
        return adv, adv

    _, advantages = jax.lax.scan(step, jnp.zeros_like(last_value), (deltas, not_done), reverse=True)
    return advantages, advantages + values


DISCO_RL_COMMIT = "9059a29f7121d60948f25ef165e08e050e9399c8"
WEIGHTS_URL = ("https://raw.githubusercontent.com/google-deepmind/disco_rl/"
               f"{DISCO_RL_COMMIT}/disco_rl/update_rules/weights/disco_103.npz")
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "disco_103.npz"


def disco103_params(agent):
    """Return the Disco103 meta-parameters, downloading them on first use.

    The weights are published by the DiscoRL authors (CC-BY 4.0). They are not stored in this
    repository. `agent` is a disco_rl Agent built with the Disco settings; its randomly initialised
    rule parameters are used to check that the loaded tensors have the expected shapes.
    """
    if not WEIGHTS_PATH.exists():
        WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading Disco103 weights to {WEIGHTS_PATH}", flush=True)
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
    with open(WEIGHTS_PATH, "rb") as f:
        flat = dict(np.load(f))
    params = {}
    for key_wb in flat:
        key = "/".join(key_wb.split("/")[:-1])
        params[key] = {"b": flat[f"{key}/b"], "w": flat[f"{key}/w"]}
    reference, _ = agent.update_rule.init_params(jax.random.PRNGKey(0))
    chex.assert_trees_all_equal_shapes_and_dtypes(reference, params)
    return params
