"""The plain-JAX port of OPEN must reproduce the original implementation.

open_fixture.npz holds inputs and the parameters the original code produced from them over several
consecutive optimizer steps (see make_open_fixture.py). The test feeds the same inputs to the port,
using the released weights, and compares the parameters after every step.

  uv run python tests/test_open.py        (or: uv run pytest tests)
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import flatten_util

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import open_optimizer  # noqa: E402

WEIGHTS = {"gridworld": (16, 8), "multi": (32, 16)}  # name -> (MLP hidden size, GRU features)
STEPS = 5
SIZES = {"actor": [8, 16, 16, 4], "critic": [8, 16, 16, 1]}
TOLERANCE = 1e-5  # the updates themselves are about 1e-3 per step


def template():
    """Actor and critic parameters laid out like the original's flax networks."""
    return {net: {"params": {f"Dense_{i}": {"bias": jnp.zeros(o), "kernel": jnp.zeros((n, o))}
                             for i, (n, o) in enumerate(zip(sizes[:-1], sizes[1:]))}}
            for net, sizes in SIZES.items()}


def activation_sizes():
    return {net: sizes[1:] for net, sizes in SIZES.items()}


def test_port_matches_original():
    fixture = np.load(Path(__file__).resolve().parent / "open_fixture.npz")
    _, unravel = flatten_util.ravel_pytree(template())
    per_layer = lambda fn: {net: {"params": {f"Dense_{i}": {"bias": fn(net, i), "kernel": fn(net, i)}
                                             for i in range(3)}} for net in SIZES}
    layer_props = per_layer(lambda net, i: i / 2)
    masks = per_layer(lambda net, i: float(net == "actor"))
    worst = 0.0
    for name in WEIGHTS:
        theta = open_optimizer.load_pretrained(name)
        params = unravel(jnp.asarray(fixture[f"{name}/init"]))
        state = open_optimizer.init(theta, params)
        for step in range(STEPS):
            acts, offset, layers = fixture[f"{name}/{step}/acts"], 0, {}
            for net, sizes in activation_sizes().items():
                for i, n in enumerate(sizes):
                    layers[net, i] = open_optimizer.dormancy(jnp.asarray(acts[offset:offset + n]))
                    offset += n
            training_prop, batch_prop = fixture[f"{name}/{step}/props"]
            params, state = open_optimizer.update(
                theta, params, state, unravel(jnp.asarray(fixture[f"{name}/{step}/grads"])),
                per_layer(lambda net, i: layers[net, i]), layer_props, masks,
                jax.random.wrap_key_data(jnp.asarray(fixture[f"{name}/{step}/key"])), training_prop, batch_prop)
            diff = float(np.abs(np.asarray(flatten_util.ravel_pytree(params)[0])
                                - fixture[f"{name}/{step}/expected"]).max())
            worst = max(worst, diff)
            assert diff < TOLERANCE, f"{name}, step {step}: parameters differ from the original by {diff:.2e}"
    return worst


if __name__ == "__main__":
    print(f"port matches the original: largest parameter difference {test_port_matches_original():.2e}")
