"""The Disco103 learner must not read the acting agent's q, y, z and aux_pi outputs from a rollout.

train_jax.py leaves these out of its replay buffer and gives the learner zeros instead. This test
makes one learner step on a real batch and one on the same batch with those outputs set to zero, and
requires every number of the two results to be equal.

  uv run python tests/test_disco_unread.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train_disco  # noqa: E402
import train_jax  # noqa: E402
from disco_rl import types, utils  # noqa: E402

env = train_disco.BatchedEnv("CartPole-v1", 64, seed=0)
agent = train_disco.make_agent("disco", 3e-3, env, (64, 64), 64)
rule_params = train_disco.load_rule_params(agent, "disco")
key = jax.random.PRNGKey(0)
learner_state, actor_state = agent.initial_learner_state(key), agent.initial_actor_state(key)
_, ts = env.reset()
timesteps = []
for _ in range(29):
    key, k = jax.random.split(key)
    actor_timestep, actor_state = agent.actor_step(learner_state.params, k, ts, actor_state)
    _, ts = env.step(None, actor_timestep.actions)
    timesteps.append(actor_timestep)
batch = types.ActorRollout.from_timestep(utils.tree_stack(timesteps, axis=0))
assert set(train_jax.UNREAD_AGENT_OUTS) <= set(batch.agent_outs), sorted(batch.agent_outs)
zeroed = batch.replace(agent_outs={k: jnp.zeros_like(v) if k in train_jax.UNREAD_AGENT_OUTS else v
                                   for k, v in batch.agent_outs.items()})

step = jax.jit(agent.learner_step, static_argnums=(5,))
state_a, state_b = learner_state, learner_state
for i in range(3):  # a few consecutive steps, so that optimizer state is compared too
    k = jax.random.PRNGKey(100 + i)
    out_a = step(k, batch, state_a, actor_state, rule_params, False)
    out_b = step(k, zeroed, state_b, actor_state, rule_params, False)
    state_a, state_b = out_a[0], out_b[0]
    for a, b in zip(jax.tree.leaves(out_a), jax.tree.leaves(out_b)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
changed = max(float(jnp.abs(a - b).max()) for a, b in zip(jax.tree.leaves(state_a.params), jax.tree.leaves(learner_state.params)))
assert changed > 0, "the learner step did not change the parameters"
print(f"learner steps are identical with and without {', '.join(train_jax.UNREAD_AGENT_OUTS)} "
      f"(parameters moved by up to {changed:.2e})")
