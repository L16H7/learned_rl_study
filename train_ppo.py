"""Proximal policy optimisation (PPO, clipped objective).

Each iteration collects `num_steps` steps from every environment, computes GAE once, then runs
`epochs` passes over the data in `minibatches` shuffled minibatches. Advantages are normalised per
minibatch; the value loss is not clipped. The actor and the critic are separate networks trained
with one Adam optimizer and a global gradient-norm clip.

  uv run python train_ppo.py --out runs/ppo_lr3e-4_s0 --lr 3e-4 --seed 0
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax

import common


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--env", default=common.DEFAULT_ENV, help="a Gymnasium environment with discrete actions")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--env_steps", type=int, default=1_000_000)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--num_steps", type=int, default=64, help="rollout length per environment")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--max_episode_steps", type=int, default=None,
                   help="episode step limit; default: the environment's own limit")
    p.add_argument("--timeout_penalty", type=float, default=0.0,
                   help="subtracted from the training reward when an episode hits the step limit")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    batch = args.num_envs * args.num_steps
    if batch % args.minibatches:
        raise ValueError("num_envs * num_steps must be divisible by minibatches")
    log = common.RunLog(args.out, {**vars(args), "rule": "ppo"})

    env = common.VecEnv(args.env, args.num_envs, args.seed, args.max_episode_steps, args.timeout_penalty)
    actor, critic = common.policy_net(env.num_actions, args.hidden), common.value_net(args.hidden)
    key = jax.random.PRNGKey(args.seed)
    key, k_actor, k_critic = jax.random.split(key, 3)
    dummy = jnp.zeros((1, env.obs_dim))
    params = {"actor": actor.init(k_actor, dummy), "critic": critic.init(k_critic, dummy)}
    opt = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.lr, eps=1e-5))
    opt_state = opt.init(params)

    @jax.jit
    def act(params, obs, key):
        key, k = jax.random.split(key)
        logp = jax.nn.log_softmax(actor.apply(params["actor"], obs))
        actions = jax.random.categorical(k, logp)
        logp_a = jnp.take_along_axis(logp, actions[:, None], axis=1)[:, 0]
        return actions, logp_a, critic.apply(params["critic"], obs)[:, 0], key

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

    @jax.jit
    def update(params, opt_state, obs, actions, logp_old, values, rewards, dones, last_obs, key):
        last_value = critic.apply(params["critic"], last_obs)[:, 0]
        advantages, targets = common.gae(values, last_value, rewards, dones, args.gamma, args.gae_lambda)
        data = (obs.reshape(batch, -1), actions.reshape(batch), logp_old.reshape(batch),
                advantages.reshape(batch), targets.reshape(batch))

        def minibatch(carry, idx):
            params, opt_state = carry
            grads = jax.grad(loss_fn)(params, *[x[idx] for x in data])
            updates, opt_state = opt.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), None

        def epoch(carry, k):
            order = jax.random.permutation(k, batch).reshape(args.minibatches, -1)
            return jax.lax.scan(minibatch, carry, order)[0], None

        (params, opt_state), _ = jax.lax.scan(epoch, (params, opt_state), jax.random.split(key, args.epochs))
        return params, opt_state

    obs, updates = env.reset(), 0
    while env.total_steps < args.env_steps:
        b_obs, b_act, b_logp, b_val, b_rew, b_done = [], [], [], [], [], []
        for _ in range(args.num_steps):
            actions, logp_a, values, key = act(params, obs, key)
            actions = np.asarray(actions)
            b_obs.append(obs)
            b_act.append(actions)
            b_logp.append(np.asarray(logp_a))
            b_val.append(np.asarray(values))
            obs, rewards, dones = env.step(actions)
            b_rew.append(rewards)
            b_done.append(dones.astype(np.float32))
        key, k = jax.random.split(key)
        params, opt_state = update(params, opt_state, np.stack(b_obs), np.stack(b_act), np.stack(b_logp),
                                   np.stack(b_val), np.stack(b_rew), np.stack(b_done), obs, k)
        updates += args.epochs * args.minibatches
        log(env.total_steps, updates, env.finished, params)
    log(env.total_steps, updates, env.finished, params, final=True)


if __name__ == "__main__":
    main()
