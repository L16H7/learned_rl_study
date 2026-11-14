"""Run the benchmark for one environment: a setting search, then the chosen settings on fresh seeds.

  uv run python bench.py cartpole tune     # search every method's setting with seeds 0-2 -> runs/cartpole_tune
  uv run python bench.py cartpole final    # each method's best setting with seeds 10-19 -> runs/cartpole_final
  uv run python bench.py cartpole all      # both, one after the other
  uv run python bench.py cartpole all --groups main          # only some method groups
  uv run python bench.py cartpole all --parallel 4           # training runs at once (default 8)
  uv run python bench.py cartpole final --final_seeds 20     # more evaluation seeds (10-29)

Environments. Most are JAX environments from gymnax, trained by train_jax.py with all seeds of a
setting in one process. LunarLander is a Gymnasium environment, trained by the single-seed scripts.

Search. Every method has one searched setting: the learning rate, or for OPEN which released weights
to use (on the MinAtar games also the weights trained on that game alone). A setting's score is its
final return (mean of the last 100 episodes) averaged over the search seeds; exact ties are
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
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent
ENVS = {  # name -> (library, environment id, environment steps per run, extra arguments for every run)
    "cartpole": ("gymnax", "CartPole-v1", 500_000, []),
    "acrobot": ("gymnax", "Acrobot-v1", 500_000, []),
    "catch": ("gymnax", "Catch-bsuite", 200_000, []),
    "deepsea": ("gymnax", "DeepSea-bsuite", 300_000, []),
    "fourrooms": ("gymnax", "FourRooms-misc", 300_000, []),
    "breakout": ("gymnax", "Breakout-MinAtar", 1_500_000, []),
    "spaceinvaders": ("gymnax", "SpaceInvaders-MinAtar", 1_500_000, []),
    "asterix": ("gymnax", "Asterix-MinAtar", 1_500_000, []),
    "freeway": ("gymnax", "Freeway-MinAtar", 1_500_000, []),
    "lunarlander": ("gymnasium", "LunarLander-v3", 1_000_000, ["--max_episode_steps", "400"]),
    # the same task, but the training reward is reduced by 100 when an episode hits the step limit
    "lunarlander_penalty": ("gymnasium", "LunarLander-v3", 1_000_000,
                            ["--max_episode_steps", "400", "--timeout_penalty", "100"]),
}
LADDER = ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2", "1e-1"]  # learning rates, in order
MAX_EXTENSIONS = 2
SMALL_AGENT = ["--torso", "64", "64", "--model_size", "64"]  # Disco103 with a torso the size of the baselines'
WIDE = ["--hidden", "512", "512"]  # baselines with networks the size of Disco103's default torso
REFERENCE_RATIO = ["--replay_ratio", "32"]  # as in the disco_rl example notebook: 8 updates per 16-env rollout
LOW_RATIO = ["--replay_ratio", "4"]  # one update per rollout


class Method(NamedTuple):
    gymnasium: list  # script of the Gymnasium version, and arguments only it takes
    gymnax: str  # --method of train_jax.py
    args: list  # arguments of both versions
    flag: str  # the searched argument
    values: list  # its starting values, in order


METHODS = {
    "reinforce": Method(["train_reinforce.py"], "reinforce", [], "--lr", ["1e-3", "3e-3", "1e-2"]),
    "a2c": Method(["train_a2c.py"], "a2c", [], "--lr", ["1e-4", "3e-4", "1e-3", "3e-3"]),
    "ppo": Method(["train_ppo.py"], "ppo", [], "--lr", ["3e-4", "1e-3", "3e-3"]),
    "pporef": Method(["train_open.py", "--optimizer", "adam"], "pporef", [], "--lr", ["1e-4", "3e-4", "1e-3", "3e-3"]),
    "open": Method(["train_open.py"], "open", [], "--weights", ["gridworld", "multi"]),
    "disco64": Method(["train_disco.py"], "disco", [*SMALL_AGENT, *REFERENCE_RATIO], "--lr", ["1e-3", "3e-3", "1e-2"]),
    "disco": Method(["train_disco.py"], "disco", REFERENCE_RATIO, "--lr", ["1e-3", "3e-3", "1e-2"]),
    "disco64_rr4": Method(["train_disco.py"], "disco", [*SMALL_AGENT, *LOW_RATIO], "--lr",
                          ["1e-3", "3e-3", "1e-2", "3e-2"]),
    "disco_rr4": Method(["train_disco.py"], "disco", LOW_RATIO, "--lr", ["1e-3", "3e-3", "1e-2", "3e-2"]),
    "reinforce512": Method(["train_reinforce.py"], "reinforce", WIDE, "--lr", ["1e-4", "3e-4", "1e-3", "3e-3"]),
    "a2c512": Method(["train_a2c.py"], "a2c", WIDE, "--lr", ["1e-4", "3e-4", "1e-3"]),
    "ppo512": Method(["train_ppo.py"], "ppo", WIDE, "--lr", ["1e-4", "3e-4", "1e-3", "3e-3"]),
}
# OPEN weights meta-trained on that environment alone
EXTRA_VALUES = {(name, "open"): [name] for name in ("breakout", "spaceinvaders", "asterix", "freeway")}
GROUPS = {
    "main": ["reinforce", "a2c", "ppo", "pporef", "open", "disco64", "disco"],
    "wide": ["reinforce512", "a2c512", "ppo512"],
    "low_ratio": ["disco64_rr4", "disco_rr4"],
}
SEARCH_SEEDS = (0, 1, 2)
FIRST_FINAL_SEED = 10
PARALLEL = 8  # runs at once unless --parallel says otherwise; each run is limited to one CPU thread
SOURCES = ["bench.py", "common.py", "open_optimizer.py", "train_a2c.py", "train_disco.py", "train_jax.py",
           "train_open.py", "train_ppo.py", "train_reinforce.py", "uv.lock"]


def start_values(name, method):
    return [*METHODS[method].values, *EXTRA_VALUES.get((name, method), [])]


def run_name(method, value):
    return f"{method}_{value}" if METHODS[method].flag == "--weights" else f"{method}_lr{value}"


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


def machine():
    """The operating system and, if there is one, the NVIDIA GPU of this machine."""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True).stdout.strip().splitlines()
    except OSError:  # no nvidia-smi: no NVIDIA GPU
        gpu = []
    return {"platform": platform.platform(), "gpu": gpu[0] if gpu else None}


def launch(jobs, folder: Path, finished: int):
    """Run every job, PARALLEL at a time. A job is (log name, names of the runs it makes, command)."""
    (folder / "logs").mkdir(parents=True, exist_ok=True)
    print(f"{folder.name}: {finished} runs already finished, {sum(len(runs) for _, runs, _ in jobs)} to run",
          flush=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1",
           "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}  # lets several runs share one GPU; no effect on a CPU
    version = {**code_version(), **machine()}

    def one(job):
        log_name, runs, cmd = job
        started = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(folder / "logs" / f"{log_name}.log", "w") as log:
            code = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        if code == 0:
            for run in runs:
                (folder / run / "provenance.json").write_text(json.dumps(
                    {**version, "command": ["python", *cmd], "started": started,
                     "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
        print(f"  {'finished' if code == 0 else 'FAILED  '} {' '.join(runs)}", flush=True)
        return code

    with ThreadPoolExecutor(PARALLEL) as pool:
        failed = sum(code != 0 for code in pool.map(one, jobs))
    if failed:
        sys.exit(f"{failed} jobs failed; see {folder}/logs")


def jobs_for(name, phase, method, value, seeds):
    """(jobs that make the runs of `seeds` that have not finished yet, number of runs already finished)."""
    library, env_id, steps, env_args = ENVS[name]
    m = METHODS[method]
    folder, prefix = ROOT / "runs" / f"{name}_{phase}", run_name(method, value)
    todo = [seed for seed in seeds if scores(folder / f"{prefix}_s{seed}", steps) is None]
    common_args = ["--env", env_id, "--env_steps", str(steps), *env_args, *m.args, m.flag, value]
    out = f"runs/{name}_{phase}/{prefix}"
    if library == "gymnax":  # one process trains all seeds
        jobs = [(prefix, [f"{prefix}_s{seed}" for seed in todo],
                 ["train_jax.py", "--method", m.gymnax, "--out", out, "--seeds", *map(str, todo), *common_args])]
    else:
        jobs = [(f"{prefix}_s{seed}", [f"{prefix}_s{seed}"],
                 [*m.gymnasium, "--out", f"{out}_s{seed}", "--seed", str(seed), *common_args]) for seed in todo]
    return (jobs if todo else []), len(seeds) - len(todo)


def launch_all(name, phase, settings, seeds):
    """Run every (method, value) of `settings` with `seeds`, skipping runs that already finished."""
    jobs, finished = [], 0
    for method, value in settings:
        new, done = jobs_for(name, phase, method, value, seeds)
        jobs += new
        finished += done
    launch(jobs, ROOT / "runs" / f"{name}_{phase}", finished)


def searched(name, method):
    """setting value -> score, for every value whose search runs have all finished."""
    steps = ENVS[name][2]
    folder = ROOT / "runs" / f"{name}_tune"
    flag = METHODS[method].flag
    result = {}
    for value in (LADDER if flag == "--lr" else start_values(name, method)):
        runs = [scores(folder / f"{run_name(method, value)}_s{seed}", steps) for seed in SEARCH_SEEDS]
        if None not in runs:
            result[value] = tuple(sum(r[i] for r in runs) / len(runs) for i in (0, 1))
    return result


def tune(name, methods):
    grids = {m: start_values(name, m) for m in methods}
    for _ in range(MAX_EXTENSIONS + 1):
        launch_all(name, "tune", [(m, value) for m in methods for value in grids[m]], SEARCH_SEEDS)
        extended = False
        for m in methods:
            if METHODS[m].flag != "--lr" or len(grids[m]) >= len(METHODS[m].values) + MAX_EXTENSIONS:
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
        flag = METHODS[m].flag
        at_edge = flag == "--lr" and best in (min(found, key=LADDER.index), max(found, key=LADDER.index))
        print(f"  {m:13s} {flag[2:]} {best:10s} return {found[best][0]:8.1f}   searched: "
              + ", ".join(f"{v} -> {s[0]:.0f}" for v, s in found.items())
              + ("   NOTE: best value is at the edge of the searched range" if at_edge else ""), flush=True)
        chosen[m] = best
    return chosen


def final(name, methods, num_seeds):
    folder = ROOT / "runs" / f"{name}_final"
    print(f"{name}: chosen settings", flush=True)
    chosen = choose(name, methods)
    launch_all(name, "final", list(chosen.items()), range(FIRST_FINAL_SEED, FIRST_FINAL_SEED + num_seeds))
    path = folder / "chosen.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**previous, **chosen}, indent=1))


def main():
    global PARALLEL
    p = argparse.ArgumentParser()
    p.add_argument("env", choices=sorted(ENVS))
    p.add_argument("phase", choices=["tune", "final", "all"])
    p.add_argument("--groups", nargs="+", default=list(GROUPS), choices=sorted(GROUPS), help="method groups to run")
    p.add_argument("--methods", nargs="+", default=None, choices=sorted(METHODS), help="run only these methods")
    p.add_argument("--final_seeds", type=int, default=10, help="number of evaluation seeds, starting at 10")
    p.add_argument("--parallel", type=int, default=PARALLEL, help="training runs at once")
    args = p.parse_args()
    PARALLEL = args.parallel
    methods = args.methods or [m for g in args.groups for m in GROUPS[g]]
    if args.phase in ("tune", "all"):
        tune(args.env, methods)
    if args.phase in ("final", "all"):
        final(args.env, methods, args.final_seeds)
    print(f"done: {args.env} {args.phase}", flush=True)


if __name__ == "__main__":
    main()
