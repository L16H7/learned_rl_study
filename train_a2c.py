"""Classic advantage actor-critic (A2C): on-policy, synchronous environments,
short rollouts, one gradient step per rollout.

  loss = -mean(A * log pi(a|s)) + value_coef * 0.5 * mean((R - V(s))^2) - entropy_coef * mean(H[pi])

Advantages A and value targets R come from GAE; advantages are not normalised unless --normalize_adv is given. The actor and the critic are
separate networks trained with one optimizer (Adam by default) and a global gradient-norm clip.

  uv run python train_a2c.py --out runs/a2c_lr7e-4_s0 --lr 7e-4 --seed 0
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
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--env_steps", type=int, default=1_000_000)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--num_steps", type=int, default=8, help="rollout length per environment")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--optimizer", default="adam", choices=["adam", "rmsprop"],
                   help="rmsprop (decay 0.99, eps 1e-5) is the optimizer of the original A2C")
    p.add_argument("--normalize_adv", action="store_true",
                   help="standardise advantages within each batch (PPO does this; classic A2C does not)")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--max_episode_steps", type=int, default=None,
                   help="episode step limit; default: the environment's own limit")
    p.add_argument("--timeout_penalty", type=float, default=0.0,
                   help="subtracted from the training reward when an episode hits the step limit")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    log = common.RunLog(args.out, {**vars(args), "rule": "a2c"})

    env = common.VecEnv(args.env, args.num_envs, args.seed, args.max_episode_steps, args.timeout_penalty)
    actor, critic = common.policy_net(env.num_actions, args.hidden), common.value_net(args.hidden)
    key = jax.random.PRNGKey(args.seed)
    key, k_actor, k_critic = jax.random.split(key, 3)
    dummy = jnp.zeros((1, env.obs_dim))
    params = {"actor": actor.init(k_actor, dummy), "critic": critic.init(k_critic, dummy)}
    base = optax.adam(args.lr, eps=1e-5) if args.optimizer == "adam" else optax.rmsprop(args.lr, decay=0.99, eps=1e-5)
    opt = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), base)
    opt_state = opt.init(params)

    @jax.jit
    def act(actor_params, obs, key):
        key, k = jax.random.split(key)
        return jax.random.categorical(k, actor.apply(actor_params, obs)), key

    def loss_fn(params, obs, actions, advantages, targets):
        logp = jax.nn.log_softmax(actor.apply(params["actor"], obs))
        logp_a = jnp.take_along_axis(logp, actions[..., None], axis=-1)[..., 0]
        entropy = -(jnp.exp(logp) * logp).sum(-1)
        values = critic.apply(params["critic"], obs)[..., 0]
        return (-(advantages * logp_a).mean()
                + args.value_coef * 0.5 * jnp.square(targets - values).mean()
                - args.entropy_coef * entropy.mean())

    @jax.jit
    def update(params, opt_state, obs, actions, rewards, dones, last_obs):
        values = critic.apply(params["critic"], obs)[..., 0]
        last_value = critic.apply(params["critic"], last_obs)[..., 0]
        advantages, targets = common.gae(values, last_value, rewards, dones, args.gamma, args.gae_lambda)
        if args.normalize_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        grads = jax.grad(loss_fn)(params, obs, actions, advantages, targets)  # targets carry no gradient
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    obs, updates = env.reset(), 0
    while env.total_steps < args.env_steps:
        b_obs, b_act, b_rew, b_done = [], [], [], []
        for _ in range(args.num_steps):
            actions, key = act(params["actor"], obs, key)
            actions = np.asarray(actions)
            b_obs.append(obs)
            b_act.append(actions)
            obs, rewards, dones = env.step(actions)
            b_rew.append(rewards)
            b_done.append(dones.astype(np.float32))
        params, opt_state = update(params, opt_state, np.stack(b_obs), np.stack(b_act), np.stack(b_rew),
                                   np.stack(b_done), obs)
        updates += 1
        log(env.total_steps, updates, env.finished, params)
    log(env.total_steps, updates, env.finished, params, final=True)


if __name__ == "__main__":
    main()
