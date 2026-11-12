"""Vanilla REINFORCE: on-policy, complete episodes, Monte Carlo reward-to-go,
no baseline, no critic, no entropy bonus, no return normalisation.

  loss = -(1 / episodes) * sum_episodes sum_t G_t * log pi(a_t | s_t),   G_t = sum_{k>=t} gamma^(k-t) r_k

Each update uses one complete episode from every environment, collected concurrently.

  uv run python train_reinforce.py --out runs/reinforce_lr1e-3_s0 --lr 1e-3 --seed 0
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
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--env_steps", type=int, default=1_000_000)
    p.add_argument("--num_envs", type=int, default=16, help="episodes per update, one per environment")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--max_episode_steps", type=int, default=None,
                   help="episode step limit; default: the environment's own limit")
    p.add_argument("--timeout_penalty", type=float, default=0.0,
                   help="subtracted from the training reward when an episode hits the step limit")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    log = common.RunLog(args.out, {**vars(args), "rule": "reinforce"})

    obs_dim, num_actions = common.env_dims(args.env)
    net = common.policy_net(num_actions, args.hidden)
    params = net.init(jax.random.PRNGKey(args.seed), jnp.zeros((1, obs_dim)))
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)
    logits_fn = jax.jit(net.apply)

    def loss_fn(params, obs, actions, returns, mask):
        logp = jax.nn.log_softmax(net.apply(params, obs))
        logp_a = jnp.take_along_axis(logp, actions[:, None], axis=1)[:, 0]
        return -(mask * returns * logp_a).sum() / args.num_envs

    @jax.jit
    def update(params, opt_state, obs, actions, returns, mask):
        grads = jax.grad(loss_fn)(params, obs, actions, returns, mask)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state

    rng = np.random.default_rng(args.seed)
    envs = [common.make_env(args.env, args.max_episode_steps) for _ in range(args.num_envs)]
    seeds = [args.seed * 1000 + i for i in range(args.num_envs)]

    def collect():
        """One complete episode per environment: lists of observations, actions, (reward, penalty)."""
        nonlocal seeds
        obs = np.stack([np.asarray(e.reset(seed=sd)[0]).reshape(-1) for e, sd in zip(envs, seeds)])
        seeds = [None] * args.num_envs  # seed each environment once, then let it continue
        data = [([], [], []) for _ in envs]
        active = np.ones(args.num_envs, dtype=bool)
        while active.any():
            logits = np.asarray(logits_fn(params, obs), dtype=np.float64)  # always the full batch
            gumbel = rng.gumbel(size=logits.shape)
            actions = np.argmax(logits + gumbel, axis=1)  # a sample from softmax(logits)
            for i in np.flatnonzero(active):
                data[i][0].append(obs[i].copy())
                data[i][1].append(int(actions[i]))
                o, r, terminated, truncated, _ = envs[i].step(int(actions[i]))
                data[i][2].append((r, args.timeout_penalty if truncated and not terminated else 0.0))
                obs[i] = np.asarray(o).reshape(-1)
                active[i] = not (terminated or truncated)
        return data

    finished, total, updates = [], 0, 0
    while total < args.env_steps:
        b_obs, b_act, b_ret = [], [], []
        for e_obs, e_act, e_rew in collect():
            raw = np.array([r for r, _ in e_rew])
            train = raw - np.array([pen for _, pen in e_rew])  # the penalty affects learning only
            G, running = np.zeros(len(train)), 0.0
            for t in reversed(range(len(train))):
                running = train[t] + args.gamma * running
                G[t] = running
            total += len(raw)
            finished.append((float(raw.sum()), len(raw), total))  # logged returns use the original reward
            b_obs += e_obs
            b_act += e_act
            b_ret += list(G)
        n = len(b_act)
        size = 1 << (n - 1).bit_length()  # pad to a power of two: few recompilations
        pad = size - n
        params, opt_state = update(
            params, opt_state,
            jnp.asarray(np.concatenate([np.stack(b_obs), np.zeros((pad, obs_dim))]), dtype=jnp.float32),
            jnp.asarray(np.concatenate([b_act, np.zeros(pad, dtype=int)]), dtype=jnp.int32),
            jnp.asarray(np.concatenate([b_ret, np.zeros(pad)]), dtype=jnp.float32),
            jnp.asarray(np.concatenate([np.ones(n), np.zeros(pad)]), dtype=jnp.float32))
        updates += 1
        log(total, updates, finished, params)
    log(total, updates, finished, params, final=True)


if __name__ == "__main__":
    main()
