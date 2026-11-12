"""PPO with the learned optimizer OPEN in place of Adam (--optimizer open), or
with Adam in exactly the same training loop (--optimizer adam) as the matched reference.

The PPO loop follows the one OPEN was meta-trained in (rl_optimizer/train.py of the OPEN repository):
separate actor and critic networks, GAE, advantages normalised per minibatch, optional clipped value
loss, and the actor's and critic's gradients clipped separately before they reach the optimizer.

OPEN has no learning rate. It is told how far training has progressed, so the number of rollouts is
fixed in advance from --env_steps.

  uv run python train_open.py --out runs/open_gridworld_s0 --weights gridworld --seed 0
  uv run python train_open.py --out runs/pporef_lr1e-3_s0 --optimizer adam --lr 1e-3 --seed 0
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax

import common
import open_optimizer

ACTIVATIONS = {"tanh": jnp.tanh, "relu": jax.nn.relu}


def init_net(key, sizes, head_scale):
    """A plain MLP as {"layer0": {"w", "b"}, ...}; orthogonal weights, zero biases."""
    keys = jax.random.split(key, len(sizes) - 1)
    scales = [np.sqrt(2)] * (len(sizes) - 2) + [head_scale]
    return {f"layer{i}": {"w": jax.nn.initializers.orthogonal(s)(k, (n, o)), "b": jnp.zeros(o)}
            for i, (k, n, o, s) in enumerate(zip(keys, sizes[:-1], sizes[1:], scales))}


def forward(net, x, activation):
    """Returns the output and, for every layer, the mean absolute output of each of its neurons over
    the batch (after the activation function for hidden layers), which OPEN turns into dormancy."""
    stats = {}
    last = len(net) - 1
    for i in range(len(net)):
        x = x @ net[f"layer{i}"]["w"] + net[f"layer{i}"]["b"]
        if i < last:
            x = activation(x)
        stats[f"layer{i}"] = jnp.mean(jnp.abs(x), axis=0)
    return x, stats


def clip_by_global_norm(tree, max_norm):
    norm = optax.global_norm(tree)
    return jax.tree.map(lambda g: g * jnp.minimum(1.0, max_norm / jnp.maximum(norm, 1e-12)), tree)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--env", default=common.DEFAULT_ENV, help="a Gymnasium environment with discrete actions")
    p.add_argument("--optimizer", default="open", choices=["open", "adam"])
    p.add_argument("--weights", default="gridworld", choices=sorted(open_optimizer.PRETRAINED),
                   help="which released OPEN weights to use")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam only")
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
    p.add_argument("--no_value_clip", action="store_true",
                   help="plain squared-error value loss instead of the clipped one OPEN was trained with")
    p.add_argument("--reward_scale", type=float, default=1.0, help="training rewards are multiplied by this")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--activation", default="tanh", choices=sorted(ACTIVATIONS))
    p.add_argument("--max_episode_steps", type=int, default=None,
                   help="episode step limit; default: the environment's own limit")
    p.add_argument("--timeout_penalty", type=float, default=0.0,
                   help="subtracted from the training reward when an episode hits the step limit")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    batch = args.num_envs * args.num_steps
    if batch % args.minibatches:
        raise ValueError("num_envs * num_steps must be divisible by minibatches")
    num_rollouts = args.env_steps // batch
    use_open = args.optimizer == "open"
    log = common.RunLog(args.out, {**vars(args), "rule": "open" if use_open else "ppo_ref"})

    env = common.VecEnv(args.env, args.num_envs, args.seed, args.max_episode_steps, args.timeout_penalty)
    activation = ACTIVATIONS[args.activation]
    key = jax.random.PRNGKey(args.seed)
    key, k_actor, k_critic = jax.random.split(key, 3)
    params = {"actor": init_net(k_actor, [env.obs_dim, *args.hidden, env.num_actions], 0.01),
              "critic": init_net(k_critic, [env.obs_dim, *args.hidden, 1], 1.0)}
    depth = len(args.hidden) + 1
    like_params = lambda fn: {net: {f"layer{i}": {"w": fn(net, i), "b": fn(net, i)} for i in range(depth)}
                              for net in ("actor", "critic")}
    layer_props = like_params(lambda net, i: i / (depth - 1))  # 0 for the first layer, 1 for the last
    noise_masks = like_params(lambda net, i: float(net == "actor"))  # learned noise for the actor only

    if use_open:
        theta = open_optimizer.load_pretrained(args.weights)
        opt_state = open_optimizer.init(theta, params)
    else:
        adam = optax.adam(args.lr, eps=1e-5)
        opt_state = adam.init(params)

    @jax.jit
    def act(params, obs, key):
        key, k = jax.random.split(key)
        logp = jax.nn.log_softmax(forward(params["actor"], obs, activation)[0])
        actions = jax.random.categorical(k, logp)
        logp_a = jnp.take_along_axis(logp, actions[:, None], axis=1)[:, 0]
        return actions, logp_a, forward(params["critic"], obs, activation)[0][:, 0], key

    def loss_fn(params, obs, actions, logp_old, values_old, advantages, targets):
        logits, actor_stats = forward(params["actor"], obs, activation)
        values, critic_stats = forward(params["critic"], obs, activation)
        values = values[:, 0]
        logp = jax.nn.log_softmax(logits)
        logp_a = jnp.take_along_axis(logp, actions[:, None], axis=1)[:, 0]
        entropy = -(jnp.exp(logp) * logp).sum(-1).mean()

        value_loss = jnp.square(values - targets)
        if not args.no_value_clip:
            clipped = values_old + jnp.clip(values - values_old, -args.clip_eps, args.clip_eps)
            value_loss = jnp.maximum(value_loss, jnp.square(clipped - targets))
        value_loss = 0.5 * value_loss.mean()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ratio = jnp.exp(logp_a - logp_old)
        clipped_ratio = jnp.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
        policy_loss = -jnp.minimum(ratio * advantages, clipped_ratio * advantages).mean()
        total = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
        return total, {"actor": actor_stats, "critic": critic_stats}

    @jax.jit
    def update(params, opt_state, obs, actions, logp_old, values, rewards, dones, last_obs, rollout, key):
        last_value = forward(params["critic"], last_obs, activation)[0][:, 0]
        advantages, targets = common.gae(values, last_value, rewards, dones, args.gamma, args.gae_lambda)
        data = (obs.reshape(batch, -1), actions.reshape(batch), logp_old.reshape(batch), values.reshape(batch),
                advantages.reshape(batch), targets.reshape(batch))

        def minibatch(carry, x):
            params, opt_state = carry
            idx, k, epoch = x
            grads, stats = jax.grad(loss_fn, has_aux=True)(params, *[d[idx] for d in data])
            grads = {net: clip_by_global_norm(grads[net], args.max_grad_norm) for net in grads}
            if use_open:
                dormancies = {net: {layer: {"w": open_optimizer.dormancy(a), "b": open_optimizer.dormancy(a)}
                                    for layer, a in stats[net].items()} for net in stats}
                params, opt_state = open_optimizer.update(
                    theta, params, opt_state, grads, dormancies, layer_props, noise_masks, k,
                    training_prop=rollout / (num_rollouts - 1), batch_prop=epoch / (args.epochs - 1))
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

        (params, opt_state), _ = jax.lax.scan(
            epoch, (params, opt_state), (jax.random.split(key, args.epochs), jnp.arange(args.epochs)))
        return params, opt_state

    obs, updates = env.reset(), 0
    for rollout in range(num_rollouts):
        b_obs, b_act, b_logp, b_val, b_rew, b_done = [], [], [], [], [], []
        for _ in range(args.num_steps):
            actions, logp_a, values, key = act(params, obs, key)
            actions = np.asarray(actions)
            b_obs.append(obs)
            b_act.append(actions)
            b_logp.append(np.asarray(logp_a))
            b_val.append(np.asarray(values))
            obs, rewards, dones = env.step(actions)
            b_rew.append(rewards * args.reward_scale)
            b_done.append(dones.astype(np.float32))
        key, k = jax.random.split(key)
        params, opt_state = update(params, opt_state, np.stack(b_obs), np.stack(b_act), np.stack(b_logp),
                                   np.stack(b_val), np.stack(b_rew), np.stack(b_done), obs,
                                   jnp.asarray(rollout, dtype=jnp.float32), k)
        updates += args.epochs * args.minibatches
        log(env.total_steps, updates, env.finished, params)
    log(env.total_steps, updates, env.finished, params, final=True)


if __name__ == "__main__":
    main()
