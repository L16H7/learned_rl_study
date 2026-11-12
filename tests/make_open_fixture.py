"""Regenerate tests/open_fixture.npz by running the ORIGINAL OPEN implementation.

Needs a checkout of https://github.com/AlexGoldie/rl-learned-optimization (with its
learned_optimization submodule) and an environment with flax and gin-config. It is not needed to
run the test, only to rebuild the reference values.

  python tests/make_open_fixture.py <path to rl_optimizer in the original repository>
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import flatten_util

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, str(Path(__file__).resolve().parent))
from network import GRU_Opt  # noqa: E402  (the original implementation)
from test_open import STEPS, WEIGHTS, activation_sizes, template  # noqa: E402

rng = np.random.default_rng(0)
out = {}
for name, (hidden, gru) in WEIGHTS.items():
    flat_weights = np.load(f"{sys.argv[1]}/pretrained/{name}_OPEN.npy", allow_pickle=False)
    original = GRU_Opt(hidden_size=hidden, gru_features=gru)
    opt = original.opt_fn(flatten_util.ravel_pytree(original.init(jax.random.PRNGKey(0)))[1](flat_weights))
    zeros, unravel = flatten_util.ravel_pytree(template())
    params = unravel(jnp.asarray(rng.normal(size=zeros.shape) * 0.3, dtype=jnp.float32))
    out[f"{name}/init"] = np.asarray(flatten_util.ravel_pytree(params)[0])
    state_a = opt.init(params["actor"], key=jax.random.PRNGKey(1))
    state_c = opt.init(params["critic"], key=jax.random.PRNGKey(2))
    layer_props = {k: {"params": {f"Dense_{i}": {"bias": i / 2, "kernel": i / 2} for i in range(3)}} for k in params}
    masks = {k: {"params": {f"Dense_{i}": {"bias": float(k == "actor"), "kernel": float(k == "actor")}
                            for i in range(3)}} for k in params}
    for step in range(STEPS):
        grads = np.asarray(rng.normal(size=zeros.shape) * 10.0 ** rng.uniform(-4, 0), dtype=np.float32)
        acts = np.asarray(np.abs(rng.normal(size=sum(sum(v) for v in activation_sizes().values()))), dtype=np.float32)
        key = jax.random.PRNGKey(100 + step)
        training_prop, batch_prop = step / (STEPS - 1), (step % 4) / 3.0
        activations, offset = {}, 0
        for net, sizes in activation_sizes().items():
            layers = []
            for n in sizes:
                a = jnp.asarray(acts[offset:offset + n])
                layers.append({"kernel": a, "bias": a})
                offset += n
            activations[net] = tuple(layers)
        result = opt.update(state_a, state_c, unravel(jnp.asarray(grads)), activations, key=key,
                            training_prop=jnp.asarray(training_prop), batch_prop=jnp.asarray(batch_prop),
                            config={"VISUALISE": False}, layer_props=layer_props, mask=masks)
        state_a, state_c = result[0], result[1]
        expected = flatten_util.ravel_pytree({"actor": state_a.params, "critic": state_c.params})[0]
        out[f"{name}/{step}/grads"], out[f"{name}/{step}/acts"] = grads, acts
        out[f"{name}/{step}/key"] = np.asarray(jax.random.key_data(key))
        out[f"{name}/{step}/props"] = np.asarray([training_prop, batch_prop], dtype=np.float32)
        out[f"{name}/{step}/expected"] = np.asarray(expected)

path = Path(__file__).resolve().parent / "open_fixture.npz"
np.savez_compressed(path, **out)
print(f"wrote {path} ({path.stat().st_size} bytes)")
