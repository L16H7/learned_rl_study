"""Every method of the benchmark on a JAX environment from gymnax, several seeds at once.

The environment, the agent and the update all run inside one compiled function, so a whole run stays
on the accelerator, and the seeds given with --seeds are trained side by side (jax.vmap). The methods
are the ones of train_reinforce.py, train_a2c.py, train_ppo.py, train_open.py and train_disco.py with
the same hyperparameters; only the environment and the loop around it differ.

  uv run python train_jax.py --method ppo --env CartPole-v1 --env_steps 500000 --lr 1e-3 \
      --seeds 0 1 2 --out runs/example/ppo_lr1e-3

writes runs/example/ppo_lr1e-3_s0, ..._s1 and ..._s2, each with the files a single-seed run has
(args.json, progress.csv, episodes.csv, params.pkl).

Methods: reinforce, a2c, ppo, pporef (PPO with Adam in OPEN's loop), open (PPO with the OPEN
optimizer), disco (Disco103 with a replay buffer). Reaching the environment's step limit counts as
the end of the episode, as in the other scripts.
"""
import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional

import dm_env
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dm_env import specs

import common
import open_optimizer
import train_disco
import train_open
from disco_rl import types


# ----------------------------------------------------------------------------- environment

class Env(NamedTuple):
    reset: Callable  # key -> (state, observations [N, D])
    step: Callable  # key, state, actions [N] -> (state, observations, rewards, dones)
    num_envs: int
    obs_dim: int
    num_actions: int
    max_steps: int  # the step limit of an episode


def make_env(name: str, num_envs: int, max_episode_steps=None) -> Env:
    """`num_envs` copies of a gymnax environment with flattened float observations. After the step
    that ends an episode, the observation is the first one of the next episode."""
    env, params = gymnax.make(name)
    if max_episode_steps is not None:
        params = params.replace(max_steps_in_episode=max_episode_steps)

    def reset_one(key):
        obs, state = env.reset(key, params)
        return state, obs.reshape(-1).astype(jnp.float32)

    def step_one(key, state, action):
        obs, state, reward, done, _ = env.step(key, state, action, params)
        return state, obs.reshape(-1).astype(jnp.float32), reward.astype(jnp.float32), done

    def reset(key):
        return jax.vmap(reset_one)(jax.random.split(key, num_envs))

    def step(key, state, actions):
        return jax.vmap(step_one)(jax.random.split(key, num_envs), state, actions)

    obs_dim = int(np.prod(env.observation_space(params).shape))
    return Env(reset, step, num_envs, obs_dim, env.num_actions, int(params.max_steps_in_episode))


class Actor(NamedTuple):
    """Where the environments are, and the return and length of the episodes in progress."""
    env_state: Any
    obs: jnp.ndarray
    ep_return: jnp.ndarray
    ep_length: jnp.ndarray
    key: jnp.ndarray


class Episodes(NamedTuple):
    """Episodes that ended in one iteration; entries with done=False are empty."""
    done: jnp.ndarray
    ret: jnp.ndarray
    length: jnp.ndarray
    at: jnp.ndarray  # environment steps taken in this iteration when the episode ended


class Algo(NamedTuple):
    init: Callable  # key -> state
    step: Callable  # state -> (state, Episodes, environment steps taken, updates made)
    params: Callable  # state -> the agent's parameters, for params.pkl
    steps_per_iteration: Optional[int]  # None if it varies (REINFORCE)
    iterations: Optional[int] = None  # fixed number of iterations, if not set by the step budget
    all_seeds: bool = False  # True if init and step take every seed at once (a leading seed axis)


def start(env: Env, key) -> Actor:
    k_reset, key = jax.random.split(key)
    env_state, obs = env.reset(k_reset)
    return Actor(env_state, obs, jnp.zeros(env.num_envs), jnp.zeros(env.num_envs, jnp.int32), key)


def collect(env: Env, policy: Callable, actor: Actor, num_steps: int):
    """`num_steps` steps of every environment. policy(obs, key) -> (actions, anything to keep).
    Returns the new Actor, (obs, actions, rewards, dones, kept) with a leading time axis, and Episodes."""

    def step(a: Actor, _):
        key, k_act, k_env = jax.random.split(a.key, 3)
        actions, kept = policy(a.obs, k_act)
        env_state, obs, rewards, dones = env.step(k_env, a.env_state, actions)
        ep_return, ep_length = a.ep_return + rewards, a.ep_length + 1
        out = (a.obs, actions, rewards, dones, kept, ep_return, ep_length)
        return Actor(env_state, obs, jnp.where(dones, 0.0, ep_return), jnp.where(dones, 0, ep_length), key), out

    actor, (obs, actions, rewards, dones, kept, ep_return, ep_length) = jax.lax.scan(step, actor, None, num_steps)
    # the environments of one step are counted one after the other, as common.VecEnv does
    at = jnp.arange(num_steps)[:, None] * env.num_envs + jnp.arange(env.num_envs)[None] + 1
    return actor, (obs, actions, rewards, dones, kept), Episodes(dones, ep_return, ep_length, at)


# ----------------------------------------------------------------------------- methods

def reinforce(env: Env, args) -> Algo:
    """train_reinforce.py: one complete episode from every environment per update.

    Works on all seeds at once, so that collecting episodes stops as soon as every episode of every
    seed has ended."""
    net = common.policy_net(env.num_actions, args.hidden)
    opt = optax.adam(args.lr)
    T, N = env.max_steps, env.num_envs

    def init_one(key):
        k_params, key = jax.random.split(key)
        params = net.init(k_params, jnp.zeros((1, env.obs_dim)))
        return dict(params=params, opt_state=opt.init(params), key=key)

    def loss_fn(params, obs, actions, returns, mask):
        logp = jax.nn.log_softmax(net.apply(params, obs))
        logp_a = jnp.take_along_axis(logp, actions[..., None], axis=-1)[..., 0]
        return -(mask * returns * logp_a).sum() / N

    def act_one(params, env_state, obs, key):
        key, k_act, k_env = jax.random.split(key, 3)
        actions = jax.random.categorical(k_act, net.apply(params, obs))
        env_state, next_obs, rewards, dones = env.step(k_env, env_state, actions)
        return env_state, next_obs, actions, rewards, dones, key

    def update_one(s, key, b_obs, b_act, b_rew, b_mask):
        def discount(g, r):
            g = r + args.gamma * g
            return g, g

        _, returns = jax.lax.scan(discount, jnp.zeros(N), b_rew, reverse=True)
        grads = jax.grad(loss_fn)(s["params"], b_obs, b_act, returns, b_mask)
        updates, opt_state = opt.update(grads, s["opt_state"])
        lengths = b_mask.sum(0).astype(jnp.int32)
        episodes = Episodes(jnp.ones(N, bool), b_rew.sum(0), lengths, jnp.cumsum(lengths))
        return dict(params=optax.apply_updates(s["params"], updates), opt_state=opt_state, key=key), episodes, \
            lengths.sum(), jnp.ones((), jnp.int32)

    def step(s):  # every array has a leading seed axis
        seeds = s["key"].shape[0]
        keys = jax.vmap(lambda k: jax.random.split(k, 3))(s["key"])
        env_state, obs = jax.vmap(env.reset)(keys[:, 1])
        empty = (jnp.zeros((T, seeds, N, env.obs_dim)), jnp.zeros((T, seeds, N), jnp.int32),
                 jnp.zeros((T, seeds, N)), jnp.zeros((T, seeds, N)))

        def running(c):
            return c[0].any() & (c[1] < T)

        def one_step(c):
            active, t, env_state, obs, k, (b_obs, b_act, b_rew, b_mask) = c
            env_state, next_obs, actions, rewards, dones, k = jax.vmap(act_one)(s["params"], env_state, obs, k)
            mask = active.astype(jnp.float32)
            data = (b_obs.at[t].set(obs), b_act.at[t].set(actions), b_rew.at[t].set(rewards * mask),
                    b_mask.at[t].set(mask))
            return active & ~dones, t + 1, env_state, next_obs, k, data

        data = jax.lax.while_loop(running, one_step, (jnp.ones((seeds, N), bool), 0, env_state, obs, keys[:, 2],
                                                      empty))[5]
        return jax.vmap(update_one, in_axes=(0, 0, 1, 1, 1, 1))(s, keys[:, 0], *data)

    return Algo(jax.vmap(init_one), step, lambda s: s["params"], None, all_seeds=True)


def actor_critic_nets(env: Env, args, key):
    actor, critic = common.policy_net(env.num_actions, args.hidden), common.value_net(args.hidden)
    k_actor, k_critic = jax.random.split(key)
    dummy = jnp.zeros((1, env.obs_dim))
    return actor, critic, {"actor": actor.init(k_actor, dummy), "critic": critic.init(k_critic, dummy)}


def a2c(env: Env, args) -> Algo:
    """train_a2c.py with its defaults: Adam, advantages not normalised."""
    actor, critic = common.policy_net(env.num_actions, args.hidden), common.value_net(args.hidden)
    opt = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.lr, eps=1e-5))

    def init(key):
        k_params, k_env = jax.random.split(key)
        params = actor_critic_nets(env, args, k_params)[2]
        return dict(params=params, opt_state=opt.init(params), actor=start(env, k_env))

    def loss_fn(params, obs, actions, advantages, targets):
        logp = jax.nn.log_softmax(actor.apply(params["actor"], obs))
        logp_a = jnp.take_along_axis(logp, actions[..., None], axis=-1)[..., 0]
        entropy = -(jnp.exp(logp) * logp).sum(-1)
        values = critic.apply(params["critic"], obs)[..., 0]
        return (-(advantages * logp_a).mean() + args.value_coef * 0.5 * jnp.square(targets - values).mean()
                - args.entropy_coef * entropy.mean())

    def step(s):
        params = s["params"]
        policy = lambda obs, k: (jax.random.categorical(k, actor.apply(params["actor"], obs)), None)
        act, (obs, actions, rewards, dones, _), episodes = collect(env, policy, s["actor"], args.num_steps)
        values = critic.apply(params["critic"], obs)[..., 0]
        last_value = critic.apply(params["critic"], act.obs)[..., 0]
        advantages, targets = common.gae(values, last_value, rewards, dones.astype(jnp.float32), args.gamma,
                                         args.gae_lambda)
        grads = jax.grad(loss_fn)(params, obs, actions, advantages, targets)
        updates, opt_state = opt.update(grads, s["opt_state"], params)
        return dict(params=optax.apply_updates(params, updates), opt_state=opt_state, actor=act), episodes, \
            args.num_steps * env.num_envs, 1

    return Algo(init, step, lambda s: s["params"], args.num_steps * env.num_envs)


def ppo(env: Env, args) -> Algo:
    """train_ppo.py."""
    actor, critic = common.policy_net(env.num_actions, args.hidden), common.value_net(args.hidden)
    opt = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.lr, eps=1e-5))
    batch = env.num_envs * args.num_steps

    def init(key):
        k_params, k_env, key = jax.random.split(key, 3)
        params = actor_critic_nets(env, args, k_params)[2]
        return dict(params=params, opt_state=opt.init(params), actor=start(env, k_env), key=key)

    def loss_fn(params, obs, actions, logp_old, advantages, targets):
        logp = jax.nn.log_softmax(actor.apply(params["actor"], obs))
        logp_a = jnp.take_along_axis(logp, actions[:, None], axis=1)[:, 0]
        entropy = -(jnp.exp(logp) * logp).sum(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ratio = jnp.exp(logp_a - logp_old)
        clipped = jnp.clip(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
        policy_loss = -jnp.minimum(ratio * advantages, clipped * advantages).mean()
        values = critic.apply(params["critic"], obs)[:, 0]
        return (policy_loss + args.value_coef * 0.5 * jnp.square(targets - values).mean()
                - args.entropy_coef * entropy.mean())

    def step(s):
        params = s["params"]

        def policy(obs, k):
            logp = jax.nn.log_softmax(actor.apply(params["actor"], obs))
            actions = jax.random.categorical(k, logp)
            return actions, (jnp.take_along_axis(logp, actions[:, None], axis=1)[:, 0],
                             critic.apply(params["critic"], obs)[:, 0])

        act, (obs, actions, rewards, dones, (logp_old, values)), episodes = collect(env, policy, s["actor"],
                                                                                   args.num_steps)
        last_value = critic.apply(params["critic"], act.obs)[:, 0]
        advantages, targets = common.gae(values, last_value, rewards, dones.astype(jnp.float32), args.gamma,
                                         args.gae_lambda)
        data = (obs.reshape(batch, -1), actions.reshape(batch), logp_old.reshape(batch), advantages.reshape(batch),
                targets.reshape(batch))

        def minibatch(carry, idx):
            params, opt_state = carry
            grads = jax.grad(loss_fn)(params, *[x[idx] for x in data])
            updates, opt_state = opt.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), None

        def epoch(carry, k):
            order = jax.random.permutation(k, batch).reshape(args.minibatches, -1)
            return jax.lax.scan(minibatch, carry, order)[0], None

        key, k = jax.random.split(s["key"])
        (params, opt_state), _ = jax.lax.scan(epoch, (params, s["opt_state"]), jax.random.split(k, args.epochs))
        return dict(params=params, opt_state=opt_state, actor=act, key=key), episodes, batch, \
            args.epochs * args.minibatches

    return Algo(init, step, lambda s: s["params"], batch)


def open_loop(env: Env, args, use_open: bool) -> Algo:
    """train_open.py: PPO in the loop OPEN was meta-trained in, with OPEN or with Adam."""
    batch = env.num_envs * args.num_steps
    num_rollouts = args.env_steps // batch
    activation = train_open.ACTIVATIONS[args.activation]
    forward = train_open.forward
    depth = len(args.hidden) + 1
    like_params = lambda fn: {net: {f"layer{i}": {"w": fn(net, i), "b": fn(net, i)} for i in range(depth)}
                              for net in ("actor", "critic")}
    layer_props = like_params(lambda net, i: i / (depth - 1))
    noise_masks = like_params(lambda net, i: float(net == "actor"))
    theta = open_optimizer.load_pretrained(args.weights) if use_open else None
    adam = optax.adam(args.lr, eps=1e-5)

    def init(key):
        k_actor, k_critic, k_env, key = jax.random.split(key, 4)
        params = {"actor": train_open.init_net(k_actor, [env.obs_dim, *args.hidden, env.num_actions], 0.01),
                  "critic": train_open.init_net(k_critic, [env.obs_dim, *args.hidden, 1], 1.0)}
        opt_state = open_optimizer.init(theta, params) if use_open else adam.init(params)
        return dict(params=params, opt_state=opt_state, actor=start(env, k_env), key=key,
                    rollout=jnp.zeros((), jnp.float32))

    def loss_fn(params, obs, actions, logp_old, values_old, advantages, targets):
        logits, actor_stats = forward(params["actor"], obs, activation)
        values, critic_stats = forward(params["critic"], obs, activation)
        values = values[:, 0]
        logp = jax.nn.log_softmax(logits)
        logp_a = jnp.take_along_axis(logp, actions[:, None], axis=1)[:, 0]
        entropy = -(jnp.exp(logp) * logp).sum(-1).mean()
        clipped = values_old + jnp.clip(values - values_old, -args.clip_eps, args.clip_eps)
        value_loss = 0.5 * jnp.maximum(jnp.square(values - targets), jnp.square(clipped - targets)).mean()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ratio = jnp.exp(logp_a - logp_old)
        clipped_ratio = jnp.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
        policy_loss = -jnp.minimum(ratio * advantages, clipped_ratio * advantages).mean()
        total = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
        return total, {"actor": actor_stats, "critic": critic_stats}

    def step(s):
        params = s["params"]

        def policy(obs, k):
            logp = jax.nn.log_softmax(forward(params["actor"], obs, activation)[0])
            actions = jax.random.categorical(k, logp)
            return actions, (jnp.take_along_axis(logp, actions[:, None], axis=1)[:, 0],
                             forward(params["critic"], obs, activation)[0][:, 0])

        act, (obs, actions, rewards, dones, (logp_old, values)), episodes = collect(env, policy, s["actor"],
                                                                                   args.num_steps)
        last_value = forward(params["critic"], act.obs, activation)[0][:, 0]
        advantages, targets = common.gae(values, last_value, rewards, dones.astype(jnp.float32), args.gamma,
                                         args.gae_lambda)
        data = (obs.reshape(batch, -1), actions.reshape(batch), logp_old.reshape(batch), values.reshape(batch),
                advantages.reshape(batch), targets.reshape(batch))

        def minibatch(carry, x):
            params, opt_state = carry
            idx, k, epoch = x
            grads, stats = jax.grad(loss_fn, has_aux=True)(params, *[d[idx] for d in data])
            grads = {net: train_open.clip_by_global_norm(grads[net], args.max_grad_norm) for net in grads}
            if use_open:
                dormancies = {net: {layer: {"w": open_optimizer.dormancy(a), "b": open_optimizer.dormancy(a)}
                                    for layer, a in stats[net].items()} for net in stats}
                params, opt_state = open_optimizer.update(
                    theta, params, opt_state, grads, dormancies, layer_props, noise_masks, k,
                    training_prop=s["rollout"] / (num_rollouts - 1), batch_prop=epoch / (args.epochs - 1))
            else:
                updates, opt_state = adam.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
            return (params, opt_state), None

        def epoch(carry, x):
            k, e = x
            k_perm, k_steps = jax.random.split(k)
            order = jax.random.permutation(k_perm, batch).reshape(args.minibatches, -1)
            keys = jax.random.split(k_steps, args.minibatches)
            return jax.lax.scan(minibatch, carry, (order, keys, jnp.full(args.minibatches, e)))[0], None

        key, k = jax.random.split(s["key"])
        (params, opt_state), _ = jax.lax.scan(
            epoch, (params, s["opt_state"]), (jax.random.split(k, args.epochs), jnp.arange(args.epochs)))
        return dict(params=params, opt_state=opt_state, actor=act, key=key, rollout=s["rollout"] + 1), episodes, \
            batch, args.epochs * args.minibatches

    return Algo(init, step, lambda s: s["params"], batch, num_rollouts)


# Outputs of the acting agent that a rollout carries but the learner never reads (it reads only the
# logits; tests/test_disco_unread.py checks this). The replay buffer leaves them out: they are 90% of a rollout.
UNREAD_AGENT_OUTS = ("q", "y", "z", "aux_pi")


def disco(env: Env, args) -> Algo:
    """train_disco.py: a disco_rl agent trained by the frozen Disco103 rule from a replay buffer."""

    class Spec:  # what disco_rl needs to know about the environment
        single_observation_spec = staticmethod(lambda: specs.Array((env.obs_dim,), np.float32, name="observation"))
        single_action_spec = staticmethod(lambda: specs.BoundedArray((), np.int32, 0, env.num_actions - 1))

    agent = train_disco.make_agent("disco", args.lr, Spec, args.torso, args.model_size)
    rule_params = train_disco.load_rule_params(agent, "disco")
    N, capacity = env.num_envs, args.buffer
    updates_per_rollout, remainder = divmod(args.replay_ratio * N, args.batch_size)
    if updates_per_rollout < 1 or remainder:
        raise ValueError("replay_ratio * num_envs must be a positive multiple of batch_size")

    def rollout(s):
        """One rollout of every environment: (state without the buffer updated, the rollout, Episodes)."""

        def one_step(c, _):
            env_state, ts, actor_state, ep_return, ep_length, key = c
            key, k_act, k_env = jax.random.split(key, 3)
            actor_timestep, actor_state = agent.actor_step(s["learner"].params, k_act, ts, actor_state)
            env_state, obs, rewards, dones = env.step(k_env, env_state, actor_timestep.actions)
            ts = types.EnvironmentTimestep(
                observation=obs, reward=rewards,
                step_type=jnp.where(dones, int(dm_env.StepType.LAST), int(dm_env.StepType.MID)).astype(jnp.int32))
            ep_return, ep_length = ep_return + rewards, ep_length + 1
            c = (env_state, ts, actor_state, jnp.where(dones, 0.0, ep_return), jnp.where(dones, 0, ep_length), key)
            return c, (actor_timestep, dones, ep_return, ep_length)

        carry = (s["env_state"], s["ts"], s["actor"], s["ep_return"], s["ep_length"], s["key"])
        (env_state, ts, actor_state, ep_return, ep_length, key), (timesteps, dones, rets, lengths) = jax.lax.scan(
            one_step, carry, None, args.rollout_len)
        at = jnp.arange(args.rollout_len)[:, None] * N + jnp.arange(N)[None] + 1
        s = {**s, "env_state": env_state, "ts": ts, "actor": actor_state, "ep_return": ep_return,
             "ep_length": ep_length, "key": key}
        return s, types.ActorRollout.from_timestep(timesteps), Episodes(dones, rets, lengths, at)

    def without_unread(rollout):
        return rollout.replace(agent_outs={k: v for k, v in rollout.agent_outs.items() if k not in UNREAD_AGENT_OUTS})

    def with_unread(batch):
        """The sampled batch with zeros in place of the outputs the buffer does not keep."""
        zeros = {k: jnp.zeros((args.rollout_len, args.batch_size, *shape), dtype) for k, (shape, dtype) in unread.items()}
        return batch.replace(agent_outs={**batch.agent_outs, **zeros})

    def init_agent(key):
        k_agent, k_env, key = jax.random.split(key, 3)
        env_state, obs = env.reset(k_env)
        ts = types.EnvironmentTimestep(observation=obs, reward=jnp.zeros(N, jnp.float32),
                                       step_type=jnp.full(N, int(dm_env.StepType.FIRST), jnp.int32))
        s = dict(learner=agent.initial_learner_state(k_agent), actor=agent.initial_actor_state(k_agent),
                 env_state=env_state, ts=ts, ep_return=jnp.zeros(N), ep_length=jnp.zeros(N, jnp.int32), key=key)
        return {**s, "size": jnp.zeros((), jnp.int32), "next": jnp.zeros((), jnp.int32)}

    def init(key):
        s = init_agent(key)
        example = without_unread(rollout(s)[1])  # only its shapes are used
        buffer = jax.tree.map(lambda x: jnp.zeros((x.shape[0], capacity, *x.shape[2:]), x.dtype), example)
        return {**s, "buffer": buffer}

    example = jax.eval_shape(lambda key: rollout(init_agent(key))[1], jax.random.PRNGKey(0))
    unread = {k: (v.shape[2:], v.dtype) for k, v in example.agent_outs.items() if k in UNREAD_AGENT_OUTS}

    def step(s):
        s, new, episodes = rollout(s)
        where = (s["next"] + jnp.arange(N)) % capacity
        buffer = jax.tree.map(lambda data, x: data.at[:, where].set(x), s["buffer"], without_unread(new))
        size, nxt = jnp.minimum(s["size"] + N, capacity), (s["next"] + N) % capacity
        ready = size >= args.batch_size
        oldest = jnp.where(size == capacity, nxt, 0)

        def update(carry, _):
            learner, key = carry
            key, k_sample, k_learn = jax.random.split(key, 3)
            age_rank = jax.random.randint(k_sample, (args.batch_size,), 0, jnp.maximum(size, 1))
            batch = with_unread(jax.tree.map(lambda data: data[:, (oldest + age_rank) % capacity], buffer))
            new_learner = agent.learner_step(k_learn, batch, learner, s["actor"], rule_params, False)[0]
            return (jax.tree.map(lambda a, b: jnp.where(ready, a, b), new_learner, learner), key), None

        (learner, key), _ = jax.lax.scan(update, (s["learner"], s["key"]), None, updates_per_rollout)
        s = {**s, "learner": learner, "key": key, "buffer": buffer, "size": size, "next": nxt}
        return s, episodes, args.rollout_len * N, jnp.where(ready, updates_per_rollout, 0)

    return Algo(init, step, lambda s: s["learner"].params, args.rollout_len * N)


METHODS = {"reinforce": reinforce, "a2c": a2c, "ppo": ppo, "pporef": lambda env, args: open_loop(env, args, False),
           "open": lambda env, args: open_loop(env, args, True), "disco": disco}
RULE = {"reinforce": "reinforce", "a2c": "a2c", "ppo": "ppo", "pporef": "ppo_ref", "open": "open", "disco": "disco"}
DEFAULTS = {  # the defaults of the single-environment scripts, for the arguments not given
    "reinforce": dict(num_envs=16, lr=1e-3), "a2c": dict(num_envs=16, num_steps=8, lr=7e-4),
    "ppo": dict(num_envs=16, num_steps=64, lr=3e-4), "pporef": dict(num_envs=16, num_steps=64, lr=1e-3),
    "open": dict(num_envs=16, num_steps=64, lr=None), "disco": dict(num_envs=16, lr=1e-3),
}


# ----------------------------------------------------------------------------- running and logging

class Run(NamedTuple):
    state: Any
    env_steps: jnp.ndarray
    updates: jnp.ndarray
    iteration: jnp.ndarray


def make_chunk(algo: Algo, total_steps: int):
    """A compiled function that advances every seed by as many iterations as its second argument is long."""
    step = algo.step if algo.all_seeds else jax.vmap(algo.step)

    def iteration(runs: Run, _):
        state, episodes, steps, updates = step(runs.state)
        new = Run(state, runs.env_steps + steps, runs.updates + updates, runs.iteration + 1)
        if algo.steps_per_iteration is None:  # seeds need different numbers of iterations: stop each at its budget
            finished = runs.env_steps >= total_steps
            pick = lambda a, b: jnp.where(finished.reshape(-1, *[1] * (a.ndim - 1)), a, b)
            new = jax.tree.map(pick, runs, new)
            episodes = episodes._replace(done=episodes.done & ~finished[:, None])
        return new, (episodes, runs.env_steps)

    return jax.jit(lambda runs, xs: jax.lax.scan(iteration, runs, xs))


class SeedLog:
    """The files of one seed's run: args.json, progress.csv, episodes.csv, params.pkl."""

    def __init__(self, out: Path, args: dict):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(args, indent=1))
        self.progress = open(out / "progress.csv", "w")
        self.progress.write("env_steps,updates,episodes,mean_return_last100,mean_length_last100,elapsed_s\n")
        self.episodes = open(out / "episodes.csv", "w")
        self.episodes.write("episode,return,length,env_steps\n")
        self.count, self.recent = 0, []

    def add(self, episodes: Episodes, starts, env_steps: int, updates: int, params, elapsed: float):
        """`episodes` and `starts` cover the iterations of one chunk (leading axis); starts[i] is the number of
        environment steps taken before iteration i."""
        done = np.asarray(episodes.done).reshape(len(starts), -1)
        at = (np.asarray(starts)[:, None] + np.asarray(episodes.at).reshape(len(starts), -1))[done]
        ret, length = np.asarray(episodes.ret).reshape(len(starts), -1)[done], np.asarray(episodes.length).reshape(
            len(starts), -1)[done]
        order = np.argsort(at, kind="stable")
        self.episodes.writelines(f"{self.count + i},{ret[j]:.2f},{length[j]},{at[j]}\n" for i, j in enumerate(order))
        self.episodes.flush()
        self.count += len(order)
        self.recent = (self.recent + [(ret[j], length[j]) for j in order[-100:]])[-100:]
        mean_ret = np.mean([r for r, _ in self.recent]) if self.recent else float("nan")
        mean_len = np.mean([n for _, n in self.recent]) if self.recent else float("nan")
        self.progress.write(f"{env_steps},{updates},{self.count},{mean_ret:.1f},{mean_len:.0f},{elapsed:.0f}\n")
        self.progress.flush()
        with open(self.out / "params.pkl", "wb") as f:
            pickle.dump(params, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument("--out", required=True, help="run folders are <out>_s<seed>")
    p.add_argument("--env", default="CartPole-v1", help="a gymnax environment with discrete actions")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--env_steps", type=int, default=1_000_000, help="environment steps per seed")
    p.add_argument("--max_episode_steps", type=int, default=None,
                   help="episode step limit; default: the environment's own limit")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--num_steps", type=int, default=None, help="rollout length per environment (a2c, ppo, pporef, open)")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64], help="hidden sizes (all methods but disco)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--weights", default="gridworld", choices=sorted(open_optimizer.PRETRAINED), help="open only")
    p.add_argument("--activation", default="tanh", choices=sorted(train_open.ACTIVATIONS), help="pporef and open only")
    p.add_argument("--torso", type=int, nargs="+", default=[512, 512], help="disco: hidden sizes of the agent's torso")
    p.add_argument("--model_size", type=int, default=128, help="disco: width of the action-conditional model")
    p.add_argument("--batch_size", type=int, default=64, help="disco: trajectories sampled per update")
    p.add_argument("--rollout_len", type=int, default=29, help="disco: rollout length")
    p.add_argument("--buffer", type=int, default=1024, help="disco: replay buffer capacity in trajectories")
    p.add_argument("--replay_ratio", type=int, default=32,
                   help="disco: trajectories sampled for learning per trajectory collected")
    p.add_argument("--chunks", type=int, default=50, help="how many times progress is written during a run")
    args = p.parse_args()
    for name, value in DEFAULTS[args.method].items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    env = make_env(args.env, args.num_envs, args.max_episode_steps)
    algo = METHODS[args.method](env, args)
    chunk = make_chunk(algo, args.env_steps)
    seeds = list(args.seeds)
    logs = [SeedLog(Path(f"{args.out}_s{seed}"),
                    {**vars(args), "seed": seed, "rule": RULE[args.method], "environment_library": "gymnax"})
            for seed in seeds]
    zero = jnp.zeros(len(seeds), jnp.int32)
    keys = jnp.stack([jax.random.PRNGKey(seed) for seed in seeds])
    runs = Run(jax.jit(algo.init if algo.all_seeds else jax.vmap(algo.init))(keys), zero, zero, zero)

    if algo.steps_per_iteration is None:
        iterations, per_chunk = None, 20
    else:
        iterations = algo.iterations or -(-args.env_steps // algo.steps_per_iteration)
        per_chunk = max(1, -(-iterations // args.chunks))
    t0, done_iterations = time.time(), 0
    while True:
        length = per_chunk if iterations is None else min(per_chunk, iterations - done_iterations)
        runs, (episodes, starts) = chunk(runs, jnp.zeros(length))
        done_iterations += length
        episodes, starts, env_steps, updates = jax.device_get((episodes, starts, runs.env_steps, runs.updates))
        params = jax.device_get(algo.params(runs.state))
        for i, log in enumerate(logs):
            log.add(jax.tree.map(lambda x: x[:, i], episodes), starts[:, i], int(env_steps[i]), int(updates[i]),
                    jax.tree.map(lambda x: x[i], params), time.time() - t0)
        print(f"{int(env_steps.min())} steps, {time.time() - t0:.0f} s, mean return of the last 100 episodes: "
              + " ".join(f"{np.mean([r for r, _ in log.recent]) if log.recent else float('nan'):.1f}" for log in logs),
              flush=True)
        if (iterations is not None and done_iterations >= iterations) or \
                (iterations is None and env_steps.min() >= args.env_steps):
            break


if __name__ == "__main__":
    main()
