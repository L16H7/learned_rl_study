"""Run the benchmark for one environment: a setting search, then the chosen settings on fresh seeds.

  uv run python bench.py cartpole tune     # search every method's setting with seeds 0 and 1 -> runs/cartpole_tune
  uv run python bench.py cartpole final    # each method's best setting with seeds 10-14 -> runs/cartpole_final
  uv run python bench.py cartpole all      # both, one after the other
  uv run python bench.py cartpole all --groups main          # only some method groups
  uv run python bench.py cartpole final --final_seeds 10     # more evaluation seeds (10-19)

Search. Every method has one searched setting: the learning rate, or for OPEN which released weights
to use (on MinAtar Breakout also the weights trained on that game alone). A setting's score is its
final return (mean of the last 100 episodes) averaged over the two search seeds; exact ties are
broken by the mean return over all episodes of the run. If the best learning rate is the lowest or
highest one tried, the next value on LADDER in that direction is run too, at most MAX_EXTENSIONS
times per method.

Runs that already finished are skipped, so an interrupted benchmark can be resumed and a benchmark
can be extended with more seeds or methods later.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENVS = {  # name -> environment id, environment steps per run, extra arguments for every run
    "cartpole": ("CartPole-v1", 500_000, []),
    "acrobot": ("Acrobot-v1", 500_000, []),
    "lunarlander": ("LunarLander-v3", 1_000_000, ["--max_episode_steps", "400"]),
    # the same task, but the training reward is reduced by 100 when an episode hits the step limit
    "lunarlander_penalty": ("LunarLander-v3", 1_000_000, ["--max_episode_steps", "400", "--timeout_penalty", "100"]),
    "breakout": ("MinAtar/Breakout-v1", 1_000_000, ["--max_episode_steps", "1000"]),
}
LADDER = ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2", "1e-1"]  # learning rates, in order
MAX_EXTENSIONS = 2
SMALL_AGENT = ["--torso", "64", "64", "--model_size", "64"]  # Disco103 with a torso the size of the baselines'
WIDE = ["--hidden", "512", "512"]  # baselines with networks the size of Disco103's default torso
REFERENCE_RATIO = ["--replay_ratio", "32"]  # as in the disco_rl example notebook: 8 updates per 16-env rollout
LOW_RATIO = ["--replay_ratio", "4"]  # one update per rollout
METHODS = {  # method -> (script, fixed arguments, the searched argument, its starting values in order)
    "reinforce": ("train_reinforce.py", [], "--lr", ["1e-3", "3e-3", "1e-2"]),
    "a2c": ("train_a2c.py", [], "--lr", ["1e-4", "3e-4", "1e-3", "3e-3"]),
    "ppo": ("train_ppo.py", [], "--lr", ["3e-4", "1e-3", "3e-3"]),
    "pporef": ("train_open.py", ["--optimizer", "adam"], "--lr", ["1e-4", "3e-4", "1e-3", "3e-3"]),
    "open": ("train_open.py", [], "--weights", ["gridworld", "multi"]),
    "disco64": ("train_disco.py", [*SMALL_AGENT, *REFERENCE_RATIO], "--lr", ["1e-3", "3e-3", "1e-2"]),
    "disco": ("train_disco.py", REFERENCE_RATIO, "--lr", ["1e-3", "3e-3", "1e-2"]),
    "disco64_rr4": ("train_disco.py", [*SMALL_AGENT, *LOW_RATIO], "--lr", ["1e-3", "3e-3", "1e-2", "3e-2"]),
    "disco_rr4": ("train_disco.py", LOW_RATIO, "--lr", ["1e-3", "3e-3", "1e-2", "3e-2"]),
    "reinforce512": ("train_reinforce.py", WIDE, "--lr", ["1e-4", "3e-4", "1e-3", "3e-3"]),
    "a2c512": ("train_a2c.py", WIDE, "--lr", ["1e-4", "3e-4", "1e-3"]),
    "ppo512": ("train_ppo.py", WIDE, "--lr", ["1e-4", "3e-4", "1e-3", "3e-3"]),
}
EXTRA_VALUES = {("breakout", "open"): ["breakout"]}  # OPEN weights meta-trained on that environment alone
GROUPS = {
    "main": ["reinforce", "a2c", "ppo", "pporef", "open", "disco64", "disco"],
    "wide": ["reinforce512", "a2c512", "ppo512"],
    "low_ratio": ["disco64_rr4", "disco_rr4"],
}
SEARCH_SEEDS = (0, 1)
FIRST_FINAL_SEED = 10
PARALLEL = 8  # runs at once; each run is limited to one thread
SOURCES = ["bench.py", "common.py", "open_optimizer.py", "train_a2c.py", "train_disco.py", "train_open.py",
           "train_ppo.py", "train_reinforce.py", "uv.lock"]


def start_values(name, method):
    return [*METHODS[method][3], *EXTRA_VALUES.get((name, method), [])]


def run_name(method, value):
    return f"{method}_{value}" if METHODS[method][2] == "--weights" else f"{method}_lr{value}"


def scores(folder: Path, steps: int):
    """(final return, mean return over all episodes) of a run, or None if it has not finished.
    The final return is the mean over the last 100 episodes."""
    path = folder / "episodes.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    if len(rows) < 100 or int(rows[-1]["env_steps"]) < 0.98 * steps:
        return None
    returns = [float(r["return"]) for r in rows]
    return sum(returns[-100:]) / 100, sum(returns) / len(returns)


def code_version():
    """A hash of the source files and the lock file, and the git commit if there is one."""
    digest = hashlib.sha256()
    for name in SOURCES:
        digest.update((ROOT / name).read_bytes())
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {"source_sha256": digest.hexdigest(), "git_commit": git.stdout.strip() if git.returncode == 0 else None}


def launch(jobs, folder: Path, steps: int):
    """Run every (name, command) that has not finished yet, PARALLEL at a time."""
    (folder / "logs").mkdir(parents=True, exist_ok=True)
    todo = [(name, cmd) for name, cmd in jobs if scores(folder / name, steps) is None]
    print(f"{folder.name}: {len(jobs) - len(todo)} runs already finished, {len(todo)} to run", flush=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1",
           "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"}
    version = code_version()

    def one(job):
        name, cmd = job
        started = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(folder / "logs" / f"{name}.log", "w") as log:
            code = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        if code == 0:
            (folder / name / "provenance.json").write_text(json.dumps(
                {**version, "command": ["python", *cmd], "started": started,
                 "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
        print(f"  {'finished' if code == 0 else 'FAILED  '} {name}", flush=True)
        return code

    with ThreadPoolExecutor(PARALLEL) as pool:
        failed = sum(code != 0 for code in pool.map(one, todo))
    if failed:
        sys.exit(f"{failed} runs failed; see {folder}/logs")


def job(name, phase, method, value, seed):
    """(run name, command relative to the repository root) for one run."""
    env_id, steps, env_args = ENVS[name]
    script, fixed, flag, _ = METHODS[method]
    run = f"{run_name(method, value)}_s{seed}"
    return run, [script, "--out", f"runs/{name}_{phase}/{run}", "--seed", str(seed), "--env", env_id,
                 "--env_steps", str(steps), *env_args, *fixed, flag, value]


def searched(name, method):
    """setting value -> score, for every value whose search runs have all finished."""
    _, steps, _ = ENVS[name]
    folder = ROOT / "runs" / f"{name}_tune"
    flag = METHODS[method][2]
    result = {}
    for value in (LADDER if flag == "--lr" else start_values(name, method)):
        runs = [scores(folder / f"{run_name(method, value)}_s{seed}", steps) for seed in SEARCH_SEEDS]
        if None not in runs:
            result[value] = tuple(sum(r[i] for r in runs) / len(runs) for i in (0, 1))
    return result


def tune(name, methods):
    _, steps, _ = ENVS[name]
    folder = ROOT / "runs" / f"{name}_tune"
    grids = {m: start_values(name, m) for m in methods}
    for _ in range(MAX_EXTENSIONS + 1):
        launch([job(name, "tune", m, value, seed) for m in methods for value in grids[m] for seed in SEARCH_SEEDS],
               folder, steps)
        extended = False
        for m in methods:
            if METHODS[m][2] != "--lr" or len(grids[m]) >= len(METHODS[m][3]) + MAX_EXTENSIONS:
                continue
            found = searched(name, m)
            tried = [v for v in LADDER if v in found]  # includes values from earlier invocations
            i = LADDER.index(max(found, key=found.get))
            for edge, step in ((tried[0], -1), (tried[-1], 1)):
                if LADDER[i] == edge and 0 <= i + step < len(LADDER):
                    grids[m].append(LADDER[i + step])
                    print(f"  {m}: best learning rate {edge} is at the edge of the search; adding {LADDER[i + step]}",
                          flush=True)
                    extended = True
        if not extended:
            break


def choose(name, methods):
    """method -> the searched value with the best score."""
    chosen = {}
    for m in methods:
        found = searched(name, m)
        if not found:
            sys.exit(f"no finished search runs for {m} in {name}; run the tune phase first")
        best = max(found, key=found.get)
        flag = METHODS[m][2]
        at_edge = flag == "--lr" and best in (min(found, key=LADDER.index), max(found, key=LADDER.index))
        print(f"  {m:13s} {flag[2:]} {best:10s} return {found[best][0]:8.1f}   searched: "
              + ", ".join(f"{v} -> {s[0]:.0f}" for v, s in found.items())
              + ("   NOTE: best value is at the edge of the searched range" if at_edge else ""), flush=True)
        chosen[m] = best
    return chosen


def final(name, methods, num_seeds):
    _, steps, _ = ENVS[name]
    folder = ROOT / "runs" / f"{name}_final"
    print(f"{name}: chosen settings", flush=True)
    chosen = choose(name, methods)
    seeds = range(FIRST_FINAL_SEED, FIRST_FINAL_SEED + num_seeds)
    launch([job(name, "final", m, chosen[m], seed) for m in methods for seed in seeds], folder, steps)
    path = folder / "chosen.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**previous, **chosen}, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("env", choices=sorted(ENVS))
    p.add_argument("phase", choices=["tune", "final", "all"])
    p.add_argument("--groups", nargs="+", default=list(GROUPS), choices=sorted(GROUPS), help="method groups to run")
    p.add_argument("--methods", nargs="+", default=None, choices=sorted(METHODS), help="run only these methods")
    p.add_argument("--final_seeds", type=int, default=5, help="number of evaluation seeds, starting at 10")
    args = p.parse_args()
    methods = args.methods or [m for g in args.groups for m in GROUPS[g]]
    if args.phase in ("tune", "all"):
        tune(args.env, methods)
    if args.phase in ("final", "all"):
        final(args.env, methods, args.final_seeds)
    print(f"done: {args.env} {args.phase}", flush=True)


if __name__ == "__main__":
    main()
