"""Train a fresh agent with the released Disco103 update rule, or with the
actor-critic rule from the disco_rl repository (--rule actor_critic).

The training loop follows disco_rl/colabs/eval.ipynb (CPU-environment path, replay buffer),
without pmap. The update rule's weights stay frozen; only the agent learns.

Each iteration collects one rollout of `rollout_len` steps from every environment, adds the
`num_envs` trajectories to a replay buffer, and makes replay_ratio * num_envs / batch_size updates,
each on `batch_size` trajectories sampled from the buffer. The notebook uses replay ratio 32 (2
environments, one update per iteration); with 16 environments that is 8 updates per iteration.

  uv run python train_disco.py --out runs/disco_lr1e-3_s0 --lr 1e-3 --seed 0
"""
import argparse
import collections

import dm_env
import jax
import numpy as np
import rlax
from dm_env import specs

import common
from disco_rl import agent as agent_lib
from disco_rl import types
from disco_rl import utils
from disco_rl.environments import base


class BatchedEnv(base.Environment):
    """A Gymnasium environment with the disco_rl convention: the timestep that ends an episode carries
    the final reward, step_type LAST, and the first observation of the next episode."""

    def __init__(self, env_id: str, batch_size: int, seed: int, max_episode_steps=None, timeout_penalty: float = 0.0):
        self.batch_size = batch_size
        self.vec = common.VecEnv(env_id, batch_size, seed, max_episode_steps, timeout_penalty)

    def single_observation_spec(self):
        return specs.Array((self.vec.obs_dim,), np.float32, name="observation")

    def single_action_spec(self):
        return specs.BoundedArray((), np.int32, 0, self.vec.num_actions - 1)

    def reset(self, rng_key=None):
        n = self.batch_size
        return None, types.EnvironmentTimestep(
            observation=self.vec.reset(), step_type=np.full(n, dm_env.StepType.FIRST, dtype=np.int32),
            reward=np.zeros(n, dtype=np.float32))

    def step(self, state, actions):
        obs, rewards, dones = self.vec.step(actions)
        step_type = np.where(dones, dm_env.StepType.LAST, dm_env.StepType.MID).astype(np.int32)
        return None, types.EnvironmentTimestep(observation=obs, step_type=step_type, reward=rewards)


def make_agent(rule: str, lr: float, env, torso=(512, 512), model_size: int = 128):
    """A disco_rl agent. `torso` is the MLP that embeds the observation; `model_size` is the width of
    the action-conditional model (one LSTM step per action) and of its head MLPs. The widths of the
    prediction outputs (600 and 601) are fixed by the Disco103 rule and cannot be changed."""
    settings = agent_lib.get_settings_disco() if rule == "disco" else agent_lib.get_settings_actor_critic()
    settings.net_settings.name = "mlp"
    settings.net_settings.net_args = dict(
        dense=tuple(torso), model_arch_name="lstm", head_w_init_std=1e-2,
        model_kwargs=dict(head_mlp_hiddens=(model_size,), lstm_size=model_size))
    settings.learning_rate = lr
    return agent_lib.Agent(agent_settings=settings, single_observation_spec=env.single_observation_spec(),
                           single_action_spec=env.single_action_spec(), batch_axis_name=None)


def load_rule_params(agent, rule: str):
    if rule == "disco":
        return common.disco103_params(agent)
    return agent.update_rule.init_params(jax.random.PRNGKey(0))[0]  # hand-written rules: unused placeholder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--env", default=common.DEFAULT_ENV, help="a Gymnasium environment with discrete actions")
    p.add_argument("--rule", default="disco", choices=["disco", "actor_critic"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--env_steps", type=int, default=1_000_000)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=64, help="trajectories sampled from the buffer per update")
    p.add_argument("--rollout_len", type=int, default=29)
    p.add_argument("--buffer", type=int, default=1024, help="replay buffer capacity in trajectories")
    p.add_argument("--replay_ratio", type=int, default=32,
                   help="trajectories sampled for learning per trajectory collected")
    p.add_argument("--torso", type=int, nargs="+", default=[512, 512], help="hidden sizes of the agent's MLP torso")
    p.add_argument("--model_size", type=int, default=128, help="width of the agent's action-conditional model")
    p.add_argument("--max_episode_steps", type=int, default=None,
                   help="episode step limit; default: the environment's own limit")
    p.add_argument("--timeout_penalty", type=float, default=0.0,
                   help="subtracted from the training reward when an episode hits the step limit")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    updates_per_rollout, remainder = divmod(args.replay_ratio * args.num_envs, args.batch_size)
    if updates_per_rollout < 1 or remainder:
        raise ValueError("replay_ratio * num_envs must be a positive multiple of batch_size")
    log = common.RunLog(args.out, vars(args))

    env = BatchedEnv(args.env, args.num_envs, args.seed, args.max_episode_steps, args.timeout_penalty)
    agent = make_agent(args.rule, args.lr, env, args.torso, args.model_size)
    rule_params = load_rule_params(agent, args.rule)
    actor_step = jax.jit(agent.actor_step)
    learner_step = jax.jit(agent.learner_step, static_argnums=(5,))

    key = jax.random.PRNGKey(args.seed)
    learner_state = agent.initial_learner_state(key)
    actor_state = agent.initial_actor_state(key)
    _, ts = env.reset()
    buffer = collections.deque(maxlen=args.buffer)
    np_rng = np.random.default_rng(args.seed + 17)

    updates = 0
    while env.vec.total_steps < args.env_steps:
        timesteps = []
        for _ in range(args.rollout_len):
            key, k = jax.random.split(key)
            actor_timestep, actor_state = actor_step(learner_state.params, k, ts, actor_state)
            _, ts = env.step(None, actor_timestep.actions)
            timesteps.append(actor_timestep)
        rollout = jax.device_get(types.ActorRollout.from_timestep(utils.tree_stack(timesteps, axis=0)))
        buffer.extend(rlax.tree_split_leaves(rollout, axis=1))  # one trajectory per environment

        for _ in range(updates_per_rollout if len(buffer) >= args.batch_size else 0):
            idx = np_rng.integers(len(buffer), size=args.batch_size)
            batch = utils.tree_stack([buffer[i] for i in idx], axis=1)
            key, k = jax.random.split(key)
            learner_state, _, _ = learner_step(k, batch, learner_state, actor_state, rule_params, False)
            updates += 1
        log(env.vec.total_steps, updates, env.vec.finished, learner_state.params)
    log(env.vec.total_steps, updates, env.vec.finished, learner_state.params, final=True)


if __name__ == "__main__":
    main()
