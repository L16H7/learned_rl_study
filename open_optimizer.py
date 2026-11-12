"""OPEN, a learned optimizer for reinforcement learning, as a plain-JAX port.

OPEN replaces Adam: given a gradient it returns the parameter update. A small GRU followed by an
MLP is applied to every parameter independently. Its inputs are the parameter value, the gradient,
momentum at six timescales, how far training has progressed, how far through the current batch of
data the update is, the relative depth of the parameter's layer, and the dormancy of the neuron the
parameter feeds. For the actor it also adds learned noise to the update.

Reference: Goldie, Lu, Jackson, Whiteson, Foerster. "Can Learned Optimization Make Reinforcement
Learning Less Difficult?", NeurIPS 2024. https://github.com/AlexGoldie/rl-learned-optimization
(Apache 2.0). This file is a re-implementation of `rl_optimizer/network.py` from that repository,
without its flax, gin and learned_optimization dependencies; tests/test_open.py checks it against
the original code. The released weights are downloaded on first use and are not stored here.
"""
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

DECAYS = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])  # momentum timescales
STEP_MULT = 1e-3  # scales the update
EXP_MULT = 1e-3  # scales the exponent of the update's magnitude
NUM_FEATURES = 19

OPEN_COMMIT = "b65037bd89c09f31e80c5d49f1d78cc9c6dc43e0"
WEIGHTS_URL = ("https://raw.githubusercontent.com/AlexGoldie/rl-learned-optimization/"
               f"{OPEN_COMMIT}/rl_optimizer/pretrained/{{name}}_OPEN.npy")
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights" / "open"
# name -> (MLP hidden size, GRU features); "multi" was trained with the larger network.
PRETRAINED = {"gridworld": (16, 8), "multi": (32, 16), "breakout": (16, 8), "asterix": (16, 8),
              "freeway": (16, 8), "spaceinvaders": (16, 8), "ant": (16, 8)}


def _layout(hidden: int, gru: int):
    """(name, shape) of every tensor in a flat weight vector, in file order. This is the order in
    which jax flattens the original flax parameter tree (keys sorted alphabetically)."""
    return [
        ("gru/hn/bias", (gru,)), ("gru/hn/kernel", (gru, gru)),
        ("gru/hr/kernel", (gru, gru)), ("gru/hz/kernel", (gru, gru)),
        ("gru/in/bias", (gru,)), ("gru/in/kernel", (NUM_FEATURES, gru)),
        ("gru/ir/bias", (gru,)), ("gru/ir/kernel", (NUM_FEATURES, gru)),
        ("gru/iz/bias", (gru,)), ("gru/iz/kernel", (NUM_FEATURES, gru)),
        ("mlp/dense0/bias", (hidden,)), ("mlp/dense0/kernel", (gru, hidden)),
        ("mlp/norm0/bias", (hidden,)), ("mlp/norm0/scale", (hidden,)),
        ("mlp/dense1/bias", (hidden,)), ("mlp/dense1/kernel", (hidden, hidden)),
        ("mlp/norm1/bias", (hidden,)), ("mlp/norm1/scale", (hidden,)),
        ("mlp/dense2/bias", (3,)), ("mlp/dense2/kernel", (hidden, 3)),
    ]


def unflatten(flat, hidden: int, gru: int) -> dict:
    """Flat weight vector -> {name: tensor}."""
    theta, offset = {}, 0
    for name, shape in _layout(hidden, gru):
        size = int(np.prod(shape))
        theta[name] = jnp.asarray(flat[offset:offset + size]).reshape(shape)
        offset += size
    if offset != len(flat):
        raise ValueError(f"expected {offset} weights, found {len(flat)}")
    return theta


def load_pretrained(name: str) -> dict:
    """The released OPEN weights called `name` (see PRETRAINED), downloaded on first use."""
    hidden, gru = PRETRAINED[name]
    path = WEIGHTS_DIR / f"{name}_OPEN.npy"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading OPEN weights to {path}", flush=True)
        urllib.request.urlretrieve(WEIGHTS_URL.format(name=name), path)
    return unflatten(np.load(path, allow_pickle=False), hidden, gru)


class OptState(NamedTuple):
    momentum: Any  # per tensor: shape + (6,)
    carry: Any  # per tensor: shape + (GRU features,), the GRU's memory for every parameter
    iteration: jnp.ndarray


def init(theta: dict, params) -> OptState:
    gru = theta["gru/hr/kernel"].shape[0]
    return OptState(
        momentum=jax.tree.map(lambda p: jnp.zeros(p.shape + (len(DECAYS),)), params),
        carry=jax.tree.map(lambda p: jnp.zeros(p.shape + (gru,)), params),
        iteration=jnp.asarray(0, dtype=jnp.int32))


def dormancy(mean_abs_activation):
    """Activity of each neuron relative to its layer's mean (1 = average, 0 = dormant).
    `mean_abs_activation` is the mean absolute output of every neuron of one layer over a batch."""
    a = mean_abs_activation + 1e-11
    return a / jnp.abs(a).sum(-1, keepdims=True) * a.shape[-1]


def _gru(theta, carry, x):
    def dense(name, v, bias=True):
        out = v @ theta[f"gru/{name}/kernel"]
        return out + theta[f"gru/{name}/bias"] if bias else out

    r = jax.nn.sigmoid(dense("ir", x) + dense("hr", carry, bias=False))
    z = jax.nn.sigmoid(dense("iz", x) + dense("hz", carry, bias=False))
    n = jnp.tanh(dense("in", x) + r * dense("hn", carry))
    return (1.0 - z) * n + z * carry


def _mlp(theta, x):
    for i in (0, 1):
        x = x @ theta[f"mlp/dense{i}/kernel"] + theta[f"mlp/dense{i}/bias"]
        mean = x.mean(-1, keepdims=True)
        var = jnp.maximum(jnp.square(x).mean(-1, keepdims=True) - jnp.square(mean), 0.0)
        x = (x - mean) * jax.lax.rsqrt(var + 1e-6) * theta[f"mlp/norm{i}/scale"] + theta[f"mlp/norm{i}/bias"]
        x = jax.nn.relu(x)
    return x @ theta["mlp/dense2/kernel"] + theta["mlp/dense2/bias"]


def _tensor_update(theta, p, g, momentum, carry, dorm, layer_prop, noise_mask, key, training_prop, batch_prop):
    """The raw update for one weight matrix or bias vector, and the GRU's new memory."""
    axes = tuple(range(p.ndim))
    features = jnp.concatenate([
        p[..., None],
        jnp.sign(momentum), jnp.log(jnp.abs(momentum) + 1e-13),
        jnp.sign(g)[..., None], jnp.log(jnp.abs(g) + 1e-13)[..., None],
    ], axis=-1)
    # Normalise each of these features by its root mean square over the tensor.
    features = features * jax.lax.rsqrt(1e-5 + jnp.mean(jnp.square(features), axis=axes, keepdims=True))
    # Features that are constant over the tensor, and the dormancy of each output neuron.
    constants = jnp.broadcast_to(jnp.stack([training_prop, layer_prop, batch_prop]).astype(p.dtype), p.shape + (3,))
    dorm = jnp.broadcast_to(dorm, p.shape)[..., None]
    carry = _gru(theta, carry, jnp.concatenate([features, constants, dorm], axis=-1))
    out = _mlp(theta, carry)
    update = out[..., 0] * STEP_MULT * jnp.exp(out[..., 1] * EXP_MULT)
    update = update + out[..., 2] * noise_mask * jax.random.normal(key, p.shape) * STEP_MULT
    return update, carry


def update(theta: dict, params, state: OptState, grads, dormancies, layer_props, noise_masks, key,
           training_prop, batch_prop):
    """One optimizer step: returns (new parameters, new optimizer state).

    params, grads, dormancies, layer_props and noise_masks are trees with the same structure.
      dormancies:   per tensor, dormancy() of the layer's output neurons (a vector over the last axis)
      layer_props:  per tensor, the layer's relative depth in its network, from 0 (first) to 1 (last)
      noise_masks:  per tensor, 1 to add the learned exploration noise (the actor), 0 otherwise
    training_prop: fraction of all rollouts completed, 0 to 1
    batch_prop:    fraction of the passes over the current rollout completed, 0 to 1
    """
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(key, len(leaves))
    training_prop, batch_prop = jnp.asarray(training_prop), jnp.asarray(batch_prop)
    momentum = jax.tree.map(lambda m, g: DECAYS * m + (1.0 - DECAYS) * g[..., None], state.momentum, grads)

    results = [
        _tensor_update(theta, p, g, m, c, d, jnp.asarray(lp), nm, k, training_prop, batch_prop)
        for p, g, m, c, d, lp, nm, k in zip(
            leaves, treedef.flatten_up_to(grads), treedef.flatten_up_to(momentum),
            treedef.flatten_up_to(state.carry), treedef.flatten_up_to(dormancies),
            treedef.flatten_up_to(layer_props), treedef.flatten_up_to(noise_masks), keys)
    ]
    updates = [u for u, _ in results]
    mean = jnp.concatenate([u.ravel() for u in updates]).mean()  # the update is made zero-mean overall
    new_params = treedef.unflatten([p - (u - mean) for p, u in zip(leaves, updates)])
    new_state = OptState(momentum, treedef.unflatten([c for _, c in results]), state.iteration + 1)
    return new_params, new_state
