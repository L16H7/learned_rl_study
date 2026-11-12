"""Sanity check: train a fresh agent on Catch with the released Disco103 rule.

This is disco_rl/colabs/eval.ipynb as a script (jittable Catch, replay buffer, pmap over one device).

  uv run python catch.py --rule disco --seed 0 --out runs/catch_disco_s0.csv
"""
import argparse
import collections
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import rlax

import common
from disco_rl import agent as agent_lib
from disco_rl import types
from disco_rl import utils
from disco_rl.environments import jittable_envs

parser = argparse.ArgumentParser()
parser.add_argument("--rule", default="disco", choices=["disco", "actor_critic"])
parser.add_argument("--steps", type=int, default=1000, help="learner updates")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="runs/catch.csv")
cli = parser.parse_args()
rule, num_steps, seed, out = cli.rule, cli.steps, cli.seed, cli.out
axis_name = "i"


def replicate(tree, devices):
    return jax.tree.map(lambda x: jnp.stack([jnp.asarray(x)] * len(devices)), tree)


def get_env(batch_size):
    return jittable_envs.CatchJittableEnvironment(batch_size=batch_size, env_settings=jittable_envs.get_config_catch())


env = get_env(1)
if rule == "disco":
    settings = agent_lib.get_settings_disco()
else:
    settings = agent_lib.get_settings_actor_critic()
settings.net_settings.name = "mlp"
settings.net_settings.net_args = dict(
    dense=(512, 512), model_arch_name="lstm", head_w_init_std=1e-2,
    model_kwargs=dict(head_mlp_hiddens=(128,), lstm_size=128))
settings.learning_rate = 1e-2
agent = agent_lib.Agent(agent_settings=settings, single_observation_spec=env.single_observation_spec(),
                        single_action_spec=env.single_action_spec(), batch_axis_name=axis_name)

if rule == "disco":
    rule_params = common.disco103_params(agent)
else:
    rule_params, _ = agent.update_rule.init_params(jax.random.PRNGKey(0))


def unroll_jittable_actor(params, actor_state, ts, env_state, rng, env, rollout_len, actor_step_fn, devices):
    def _single_step(carry, step_rng):
        env_state, ts, actor_state = carry
        actor_timestep, actor_state = agent.actor_step(params, step_rng, ts, actor_state)
        env_state, ts = env.step(env_state, actor_timestep.actions)
        return (env_state, ts, actor_state), actor_timestep

    (env_state, ts, actor_state), rollout = jax.lax.scan(
        _single_step, (env_state, ts, actor_state), jax.random.split(rng, rollout_len))
    return types.ActorRollout.from_timestep(rollout), actor_state, ts, env_state


def accumulate_rewards(acc, x):
    def _step(acc, x):
        r, d = x
        acc += r
        return acc * d, acc

    return jax.lax.scan(_step, acc, x)


class Buffer:
    def __init__(self, capacity, seed):
        self.buffer = collections.deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, rollout):
        self.buffer.extend(rlax.tree_split_leaves(jax.device_get(rollout), axis=2))

    def sample(self, n):
        idx = self.rng.integers(len(self.buffer), size=n)
        return utils.tree_stack([self.buffer[i] for i in idx], axis=2)

    def __len__(self):
        return len(self.buffer)


batch_size, rollout_len, replay_ratio = 64, 29, 32
rng_key = jax.random.PRNGKey(seed)
buffer = Buffer(1024, 17 + seed)
num_envs = batch_size // replay_ratio
devices = tuple(jax.devices()[:num_envs])
env = get_env(num_envs)
env_state, ts = env.reset(rng_key)
learner_state = agent.initial_learner_state(rng_key)
actor_state = agent.initial_actor_state(rng_key)

learner_step = jax.pmap(agent.learner_step, axis_name=axis_name, devices=devices, static_broadcasted_argnums=(5,))
unroll = jax.pmap(unroll_jittable_actor, axis_name=axis_name, devices=devices, static_broadcasted_argnums=(5, 6, 7, 8))
acc_fn = jax.pmap(accumulate_rewards, axis_name=axis_name, devices=devices)

learner_state = replicate(learner_state, devices)
actor_state = replicate(actor_state, devices)
rule_params = replicate(rule_params, devices)
acc = jnp.zeros((num_envs,))
env_state, ts, acc = utils.shard_across_devices((env_state, ts, acc), devices)

rows, total, t0 = [], 0, time.time()
for step in range(num_steps):
    rng_key, ra, rl = jax.random.split(rng_key, 3)
    rollout, actor_state, ts, env_state = unroll(
        learner_state.params, actor_state, ts, env_state, jax.random.split(ra, len(devices)),
        env, rollout_len, None, devices)
    buffer.add(rollout)
    total += int(np.prod(rollout.rewards.shape))
    acc, returns = acc_fn(acc, (rollout.rewards, rollout.discounts))
    returns, discounts = np.asarray(returns), np.asarray(rollout.discounts)
    ended = 1 - discounts
    rows.append((total, float((returns * ended).sum()), float(ended.sum())))
    if len(buffer) >= batch_size:
        learner_state, _, _ = learner_step(
            jax.random.split(rl, len(devices)), buffer.sample(batch_size), learner_state, actor_state,
            rule_params, False)
    if step in (0, 1, 9) or (step + 1) % 100 == 0:
        recent = rows[-100:]
        avg = sum(r[1] for r in recent) / max(1.0, sum(r[2] for r in recent))
        print(f"step {step + 1:5d}  env steps {total:6d}  avg return (last 100 steps) {avg:+.3f}  "
              f"elapsed {time.time() - t0:6.1f}s", flush=True)

Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    f.write("env_steps,return_sum,episodes\n")
    for r in rows:
        f.write(f"{r[0]},{r[1]},{r[2]}\n")
