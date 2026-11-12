"""Score a trained agent and save a GIF of a few episodes.

Works with runs from every training script here.

  uv run python watch.py runs/lunarlander_final/ppo_lr3e-3_s10 --episodes 3
"""
import argparse
import json
import pickle
from pathlib import Path

import dm_env
import jax
import numpy as np
from PIL import Image


def make_logits_fn(cfg, params):
    """Return observation -> action logits for the agent described by a run's args.json."""
    env_id = cfg.get("env", "LunarLander-v3")
    if cfg["rule"] in ("reinforce", "a2c", "ppo"):
        import common

        apply = jax.jit(common.policy_net(common.env_dims(env_id)[1], cfg["hidden"]).apply)
        actor_params = params if cfg["rule"] == "reinforce" else params["actor"]
        return lambda obs: np.asarray(apply(actor_params, obs[None]))[0]

    if cfg["rule"] in ("open", "ppo_ref"):
        import train_open

        activation = train_open.ACTIVATIONS[cfg["activation"]]
        apply = jax.jit(lambda net, obs: train_open.forward(net, obs, activation)[0])
        return lambda obs: np.asarray(apply(params["actor"], obs[None]))[0]

    import train_disco
    from disco_rl import types

    agent = train_disco.make_agent(cfg["rule"], cfg["lr"], train_disco.BatchedEnv(env_id, 1, 0),
                                   cfg.get("torso", (512, 512)), cfg.get("model_size", 128))
    actor_step = jax.jit(agent.actor_step)
    actor_state = agent.initial_actor_state(jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(0)

    def logits(obs):
        ts = types.EnvironmentTimestep(observation=obs[None].astype(np.float32),
                                       step_type=np.array([dm_env.StepType.MID], dtype=np.int32),
                                       reward=np.zeros(1, dtype=np.float32))
        out, _ = actor_step(params, key, ts, actor_state)
        return np.asarray(out.logits[0])

    return logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--episodes", type=int, default=3, help="episodes rendered into the GIF")
    p.add_argument("--eval_episodes", type=int, default=20, help="episodes for the score (not rendered)")
    p.add_argument("--greedy", action="store_true", help="take the most likely action instead of sampling")
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    run = Path(args.run)
    cfg = json.loads((run / "args.json").read_text())
    with open(run / "params.pkl", "rb") as f:
        params = pickle.load(f)
    logits_fn = make_logits_fn(cfg, params)
    rng = np.random.default_rng(args.seed)

    def episode(env, seed, frames=None):
        obs, _ = env.reset(seed=seed)
        total, done = 0.0, False
        while not done:
            logits = logits_fn(np.asarray(obs, dtype=np.float32).reshape(-1)).astype(np.float64)
            if args.greedy:
                action = int(np.argmax(logits))
            else:
                probs = np.exp(logits - logits.max())
                action = int(rng.choice(len(probs), p=probs / probs.sum()))
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            done = terminated or truncated
            if frames is not None:
                frames.append(Image.fromarray(env.render()))
        return total

    import common

    env_id, cap = cfg.get("env", "LunarLander-v3"), cfg.get("max_episode_steps")
    env = common.make_env(env_id, cap)
    scores = [episode(env, args.seed + i) for i in range(args.eval_episodes)]
    print(f"{args.eval_episodes} evaluation episodes: mean return {np.mean(scores):.1f}, "
          f"min {np.min(scores):.1f}, max {np.max(scores):.1f}, "
          f"episodes: {len(scores)}")

    env = common.make_env(env_id, cap, render_mode="rgb_array")
    frames = []
    for i in range(args.episodes):
        score = episode(env, args.seed + 1000 + i, frames)
        print(f"rendered episode {i + 1}: return {score:.1f}")
    frames = [f.resize((f.width // 2, f.height // 2)) for f in frames[::2]]
    path = run / "episodes.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=40, loop=0)
    print(f"saved {path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
