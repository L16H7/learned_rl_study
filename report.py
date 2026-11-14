"""Turn the benchmark runs into the published results: data files, figures and REPORT.md.

  uv run python report.py        # uses every run that has finished so far

Reads runs/<env>_tune and runs/<env>_final (made by bench.py) and writes

  results/<env>/runs.csv     one row per finished run, search and evaluation
  results/<env>/curves.csv   every run's learning curve on a common grid of environment steps
  results/<env>/*.png        figures
  results/dashboard.html     the learning curves as interactive charts (dashboard.py)
  REPORT.md                  tables and figures for every environment

The report states what was measured. It can be regenerated at any time; methods whose runs have not
finished are listed as pending.
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bench
from plot import GRID, INK, INK2, SURFACE, label_ends

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
POINTS = 200  # grid points of a learning curve
WINDOW = 100  # episodes in the trailing mean, the same window as the final return

NAMES = {
    "reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO", "pporef": "PPO + Adam, OPEN's loop",
    "open": "PPO + OPEN", "disco64": "Disco103, 64-wide", "disco": "Disco103, 512-wide",
    "reinforce512": "REINFORCE, 512-wide", "a2c512": "A2C, 512-wide", "ppo512": "PPO, 512-wide",
    "disco64_rr4": "Disco103, 64-wide, replay ratio 4", "disco_rr4": "Disco103, 512-wide, replay ratio 4",
}
# colour follows the method family; line style separates variants of one family
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
DASH, DOT, DASHDOT = (0, (4, 2)), (0, (1, 2)), (0, (4, 2, 1, 2))
STYLE = {
    "reinforce": (YELLOW, "solid"), "a2c": (ORANGE, "solid"), "ppo": (AQUA, "solid"), "pporef": (GREEN, "solid"),
    "open": (MAGENTA, "solid"), "disco64": (BLUE, "solid"), "disco": (BLUE, DASH),
    "reinforce512": (YELLOW, DASH), "a2c512": (ORANGE, DASH), "ppo512": (AQUA, DASH),
    "disco64_rr4": (BLUE, DOT), "disco_rr4": (BLUE, DASHDOT),
}
ENV_INFO = {  # title, success threshold on the return (or None), step limit, return-axis limits, decimals
    "catch": ("Catch (bsuite, gymnax)", 0.9, 9, (-1.05, 1.05), 2),
    "cartpole": ("CartPole-v1 (gymnax)", 475, 500, (0, 520), 0),
    "acrobot": ("Acrobot-v1 (gymnax)", -100, 500, (-520, 0), 0),
    "fourrooms": ("FourRooms (gymnax)", 0.9, 500, (-0.05, 1.05), 2),
    "deepsea": ("DeepSea, size 8 (bsuite, gymnax)", 0.9, 8, (-0.1, 1.05), 2),
    "breakout": ("MinAtar Breakout (gymnax)", None, 1000, None, 1),
    "spaceinvaders": ("MinAtar Space Invaders (gymnax)", None, 1000, None, 1),
    "asterix": ("MinAtar Asterix (gymnax)", None, 1000, None, 1),
    "freeway": ("MinAtar Freeway (gymnax)", None, 2500, None, 1),
    "lunarlander": ("LunarLander-v3 (Gymnasium), 400-step limit", 200, 400, (-400, 320), 0),
    "lunarlander_penalty": ("LunarLander-v3 (Gymnasium), 400-step limit, training reward −100 at the limit", 200,
                            400, (-400, 320), 0),
}
ENV_ORDER = [env for env in ENV_INFO if bench.ENVS[env][0] == "gymnax"]  # the report covers the gymnax runs
FIGURES = [  # file name, title, methods
    ("main.png", "main methods", bench.GROUPS["main"]),
    ("replay_ratio.png", "Disco103 at replay ratio 32 and 4", ["disco64", "disco", "disco64_rr4", "disco_rr4"]),
    ("width.png", "baselines with 64×64 and 512×512 networks",
     ["reinforce", "reinforce512", "a2c", "a2c512", "ppo", "ppo512"]),
]
RUN_NAME = re.compile(r"^(?P<method>.+?)_(?:lr(?P<lr>[0-9.e-]+)|(?P<weights>[a-z]+))_s(?P<seed>\d+)$")


def load(env):
    """One dict per finished run of `env`, from the search and the evaluation folders."""
    steps = bench.ENVS[env][2]
    _, success, limit, _, _ = ENV_INFO[env]
    grid = np.linspace(steps / POINTS, steps, POINTS)
    runs = []
    for phase in ("tune", "final"):
        folder = ROOT / "runs" / f"{env}_{phase}"
        for d in sorted(folder.iterdir()) if folder.exists() else []:
            m = RUN_NAME.match(d.name)
            if not m or m["method"] not in bench.METHODS or bench.scores(d, steps) is None:
                continue
            rows = list(csv.DictReader(open(d / "episodes.csv")))
            ret = np.array([float(r["return"]) for r in rows])
            length = np.array([int(r["length"]) for r in rows])
            at = np.array([int(r["env_steps"]) for r in rows])
            csum_r, csum_l = np.concatenate([[0], np.cumsum(ret)]), np.concatenate([[0], np.cumsum(length)])
            end = np.searchsorted(at, grid, side="right")  # episodes finished by each grid step
            start = np.maximum(end - WINDOW, 0)
            enough = end >= 10
            n = np.maximum(end - start, 1)
            trailing = np.convolve(ret, np.ones(WINDOW) / WINDOW, mode="valid")  # mean of episodes i..i+99
            reached = np.flatnonzero(trailing >= success) if success is not None else []
            provenance = d / "provenance.json"
            prov = json.loads(provenance.read_text()) if provenance.exists() else {}
            last = slice(-WINDOW, None)
            runs.append(dict(
                env=env, phase="search" if phase == "tune" else "evaluation", method=m["method"],
                setting=m["lr"] or m["weights"], seed=int(m["seed"]),
                final_return=ret[last].mean(), mean_return_all_episodes=ret.mean(),
                success_share_last100=np.mean(ret[last] >= success) if success is not None else None,
                first_step_success=int(at[reached[0] + WINDOW - 1]) if len(reached) else None,
                mean_length_last100=length[last].mean(), limit_share_last100=np.mean(length[last] >= limit),
                episodes=len(ret), env_steps=int(at[-1]),
                source_sha256=prov.get("source_sha256"), git_commit=prov.get("git_commit"),
                grid=grid, curve=np.where(enough, (csum_r[end] - csum_r[start]) / n, np.nan),
                length_curve=np.where(enough, (csum_l[end] - csum_l[start]) / n, np.nan)))
    return runs


def export(env, runs):
    out = RESULTS / env
    out.mkdir(parents=True, exist_ok=True)
    columns = ["env", "phase", "method", "setting", "seed", "final_return", "mean_return_all_episodes",
               "success_share_last100", "first_step_success", "mean_length_last100", "limit_share_last100",
               "episodes", "env_steps", "source_sha256", "git_commit"]

    def text(v):
        return "" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)

    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in runs:
            w.writerow([text(r[c]) for c in columns])
    with open(out / "curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "method", "setting", "seed", "env_steps", "mean_return_last100", "mean_length_last100"])
        for r in runs:
            for x, y, n in zip(r["grid"], r["curve"], r["length_curve"]):
                if not np.isnan(y):
                    w.writerow([r["phase"], r["method"], r["setting"], r["seed"], int(x), f"{y:.3f}", f"{n:.1f}"])


def evaluated(env, runs):
    """method -> evaluation runs of its chosen setting, ordered by seed."""
    chosen_file = ROOT / "runs" / f"{env}_final" / "chosen.json"
    chosen = json.loads(chosen_file.read_text()) if chosen_file.exists() else {}
    by_method = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r["phase"] == "evaluation":
            by_method[r["method"]][r["setting"]].append(r)
    result = {}
    for method, settings in by_method.items():
        setting = chosen.get(method)
        if setting not in settings:  # evaluation still running: take the setting with the most finished runs
            setting = max(settings, key=lambda s: len(settings[s]))
        result[method] = sorted(settings[setting], key=lambda r: r["seed"])
    return result


def fmt(x, decimals):
    return f"{x:.{decimals}f}".replace("-", "−")


def thousands(step):
    return "never" if step is None else f"{step / 1000:.0f}k"


def axes_style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def figure_means(env, groups, methods, title, path):
    """Mean learning curve of each method over its evaluation seeds."""
    env_title, success, _, ylim, _ = ENV_INFO[env]
    methods = [m for m in methods if m in groups]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    axes_style(ax)
    ends = []
    for m in methods:
        color, linestyle = STYLE[m]
        x = groups[m][0]["grid"] / 1000
        mean = np.mean([r["curve"] for r in groups[m]], axis=0)
        ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=2, solid_capstyle="round",
                label=f"{NAMES[m]} ({len(groups[m])} seeds)")
        ends.append((x[-1], mean[-1], NAMES[m]))
    xmax = groups[methods[0]][0]["grid"][-1] / 1000
    ax.set_xlim(0, xmax * 1.42)
    ax.set_xticks([t for t in ax.get_xticks() if 0 <= t <= xmax + 1])
    if ylim:
        ax.set_ylim(*ylim)
    if success is not None:
        ax.axhline(success, color=INK2, linewidth=1, zorder=1)
        ax.annotate(f"{success:g}".replace("-", "−"), (0, success), xytext=(4, 4), textcoords="offset points",
                    fontsize=9, color=INK2)
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_title(f"{env_title}: {title}\nMean return of the last {WINDOW} episodes, averaged over seeds",
                 loc="left", fontsize=11, color=INK, pad=10)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=2, frameon=False,
               fontsize=9, handlelength=2.6, columnspacing=1.8)
    fig.tight_layout(rect=(0, 0.045 * ((len(methods) + 1) // 2) + 0.02, 1, 1))
    fig.canvas.draw()
    label_ends(ax, ends)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def figure_seeds(env, groups, methods, path):
    """One panel per method: every evaluation seed, and the mean over seeds."""
    env_title, success, _, ylim, _ = ENV_INFO[env]
    methods = [m for m in methods if m in groups]
    cols = 4
    rows = -(-len(methods) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows + 0.8), facecolor=SURFACE, sharex=True, sharey=True,
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, m in zip(axes.flat, methods):
        ax.set_visible(True)
        axes_style(ax)
        color, _ = STYLE[m]
        x = groups[m][0]["grid"] / 1000
        for r in groups[m]:
            ax.plot(x, r["curve"], color=color, linewidth=0.9, alpha=0.55)
        ax.plot(x, np.mean([r["curve"] for r in groups[m]], axis=0), color=INK, linewidth=1.6)
        if success is not None:
            ax.axhline(success, color=INK2, linewidth=0.8, zorder=1)
        ax.set_title(f"{NAMES[m]}\n{setting_text(m, groups[m][0]['setting'])}", loc="left", fontsize=9, color=INK)
        if ylim:
            ax.set_ylim(*ylim)
    fig.suptitle(f"{env_title}: every evaluation seed (coloured) and the mean over seeds (black)", x=0.02, ha="left",
                 fontsize=11, color=INK)
    fig.supxlabel("Environment steps (thousands)", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def setting_text(method, setting):
    return f"{setting} weights" if bench.METHODS[method].flag == "--weights" else f"learning rate {setting}"


def evaluation_table(env, groups, methods, num_seeds):
    _, success, limit, _, d = ENV_INFO[env]
    head = ["Method", "Setting", "Seeds", "Final return, mean (min..max)", "Each seed"]
    if success is not None:
        threshold = f"{success:g}".replace("-", "−")
        head += [f"Episodes with return ≥ {threshold}", f"Step at which the trailing mean first reached {threshold}"]
    if env.startswith("lunarlander"):
        head += [f"Episodes that hit the {limit}-step limit"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    done = sorted((m for m in methods if m in groups), key=lambda m: -np.mean([r["final_return"] for r in groups[m]]))
    for m in done:
        g = groups[m]
        finals = [r["final_return"] for r in g]
        row = [NAMES[m], setting_text(m, g[0]["setting"]), f"{len(g)} of {num_seeds}" if len(g) < num_seeds else str(len(g)),
               f"{fmt(np.mean(finals), d)} ({fmt(min(finals), d)}..{fmt(max(finals), d)})",
               ", ".join(fmt(x, d) for x in finals)]
        if success is not None:
            row += [f"{np.mean([r['success_share_last100'] for r in g]):.0%}",
                    ", ".join(thousands(r["first_step_success"]) for r in g)]
        if env.startswith("lunarlander"):
            row += [f"{np.mean([r['limit_share_last100'] for r in g]):.0%}"]
        lines.append("| " + " | ".join(row) + " |")
    for m in methods:
        if m not in groups:
            lines.append(f"| {NAMES[m]} | pending |" + " |" * (len(head) - 2))
    return "\n".join(lines)


def search_table(env, runs, methods):
    _, _, _, _, d = ENV_INFO[env]
    found = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        if r["phase"] == "search":
            found[r["method"]][r["setting"]][r["seed"]] = r["final_return"]
    seeds = bench.SEARCH_SEEDS
    lines = ["| Method | Setting | " + " | ".join(f"Seed {s}" for s in seeds) + " | Mean |", "|---|---|" + "---|" * (len(seeds) + 1)]
    for m in methods:
        complete = {s: v for s, v in found[m].items() if all(seed in v for seed in seeds)}
        if not complete:
            lines.append(f"| {NAMES[m]} | pending |" + " |" * (len(seeds) + 1))
            continue
        scores = bench.searched(env, m)
        best = max(scores, key=scores.get) if scores else None
        order = bench.LADDER if bench.METHODS[m].flag == "--lr" else bench.start_values(env, m)
        for s in [s for s in order if s in complete]:
            v = complete[s]
            mark = " (chosen)" if s == best else ""
            lines.append(f"| {NAMES[m]} | {setting_text(m, s)}{mark} | " + " | ".join(fmt(v[seed], d) for seed in seeds)
                         + f" | {fmt(np.mean([v[seed] for seed in seeds]), d)} |")
    return "\n".join(lines)


INTRO = """# Report

Results of the benchmark described in [README.md](README.md): two released learned components,
Disco103 (a meta-learned update rule) and OPEN (a learned optimizer), against REINFORCE, A2C and PPO
on nine JAX environments from gymnax. This file is generated by `report.py` from the run data. It
states what was measured.

## Protocol

Each method has one searched setting: the learning rate, or for OPEN which released weights to use.
Every value is run with seeds 0, 1 and 2; if the best learning rate is the lowest or highest one
tried, the next value in that direction is run too, at most twice. The best value is then run with
ten new seeds, 10 to 19. All numbers under *Evaluation* come from those ten runs. Every method uses
16 environments and is given the same number of environment steps.

How to read the tables:

- **Final return** is the mean return of the last 100 episodes of a run. Tables give the mean over
  seeds, the lowest and highest seed, and every seed.
- **Trailing mean** is the mean return of the last 100 episodes up to a given point in training.
- With ten seeds, methods whose ranges overlap should not be ranked against each other.
- The data behind every table and figure is in `results/<environment>/runs.csv` and `curves.csv`.
  `results/dashboard.html` shows the learning curves as interactive charts.

"""

LIMITATIONS = """## Limitations

- **What was fixed before these runs.** The search grids, the step budgets and Disco103's replay
  ratio were chosen after preliminary runs of the same methods on other implementations of some of
  these tasks (Gymnasium CartPole, Acrobot, LunarLander and MinAtar Breakout). Those runs are not
  part of the results.
- **Matched on environment steps only.** The methods are not matched on the number of gradient
  updates, on compute time or on network size. Disco103's agent has prediction heads and an
  action-conditional model that the baselines do not have, whatever the torso size.
- **One searched setting per method.** All other hyperparameters are defaults: of this repository
  for the baselines, of `disco_rl` for Disco103. The search uses three seeds.
- **Chosen values at the edge of the search.** Where the table above lists a method, a value beyond
  the searched range was not tried and may be better.
- **Environments that separate the methods little.** On Catch, FourRooms and DeepSea most methods
  reach the highest possible return; they differ in how fast and in how many seeds fail. On
  DeepSea most seeds end either near 0 or near 0.99. On MinAtar Asterix no method reaches a mean final
  return of 2 within the budget.
- **Budgets.** The budgets are small compared with those the authors of Disco103 and OPEN used.
- **OPEN is not run in its own configuration.** It is the optimizer of a PPO loop with 16
  environments, 64-step rollouts and tanh networks. The weights released for each MinAtar game were
  included in that game's search. OPEN's repository pins gymnax 0.0.6; these runs use gymnax 0.0.9.
- **Episode limits.** Reaching an environment's step limit is treated as the end of the episode by
  every method: no value is bootstrapped from the state after the limit.
- **Disco103's replay buffer** does not store the agent outputs `q`, `y`, `z` and `aux_pi` of the
  acting agent; `tests/test_disco_unread.py` checks that the learner does not read them.
- **One machine, one pass.** No run was repeated on other hardware.

"""


def runs_section(all_runs):
    """What was run: counts, the code version and the machine, from the run records."""
    search = sum(r["phase"] == "search" for r in all_runs)
    hashes = sorted({r["source_sha256"] for r in all_runs if r["source_sha256"]})
    machines = set()
    for folder in sorted((ROOT / "runs").glob("*_final")):
        for path in sorted(folder.glob("*/provenance.json"))[:1]:
            prov = json.loads(path.read_text())
            machines.add((prov.get("gpu"), prov.get("platform")))
    lines = ["## Runs\n",
             f"{len(all_runs):,} training runs: {search:,} search runs and {len(all_runs) - search:,} evaluation runs.",
             "Each run's command, start and end time, code version and machine are recorded next to it "
             "(`provenance.json`); the code version is also a column of `runs.csv`.\n"]
    lines += [f"- Code version (SHA-256 over the source files and `uv.lock`, as computed by `bench.py`): `{h}`"
              for h in hashes]
    lines += [f"- Machine: {gpu or 'no GPU'}, {system}" for gpu, system in sorted(machines, key=str)]
    return "\n".join(lines) + "\n"


def edge_section():
    """Methods whose chosen learning rate is the lowest or highest value that was searched."""
    lines = ["## Chosen learning rates at the edge of the searched range\n",
             "| Environment | Method | Chosen | Searched |", "|---|---|---|---|"]
    for env in ENV_ORDER:
        for m in [m for g in bench.GROUPS.values() for m in g]:
            if bench.METHODS[m].flag != "--lr":
                continue
            found = bench.searched(env, m)
            if not found:
                continue
            tried = [v for v in bench.LADDER if v in found]
            best = max(found, key=found.get)
            if best in (tried[0], tried[-1]):
                lines.append(f"| {ENV_INFO[env][0]} | {NAMES[m]} | {best} | {tried[0]} to {tried[-1]} |")
    return "\n".join(lines) + "\n"


def main():
    parts, summary, all_runs = [INTRO], {}, []
    num_seeds = 10
    for env in ENV_ORDER:
        runs = load(env)
        title = ENV_INFO[env][0]
        if not runs:
            parts.append(f"## {title}\n\nPending.\n")
            continue
        export(env, runs)
        all_runs += runs
        groups = evaluated(env, runs)
        summary[env] = groups
        steps = bench.ENVS[env][2]
        section = [f"## {title}\n", f"{steps:,} environment steps per run, 16 environments.\n"]
        all_methods = [m for g in bench.GROUPS.values() for m in g]
        if groups:
            for name, heading, methods in (("main", "Evaluation: main methods", bench.GROUPS["main"]),
                                           ("replay_ratio", "Evaluation: Disco103 at replay ratio 4",
                                            bench.GROUPS["low_ratio"]),
                                           ("width", "Evaluation: baselines with 512×512 networks",
                                            bench.GROUPS["wide"])):
                section.append(f"### {heading}\n\n{evaluation_table(env, groups, methods, num_seeds)}\n")
                file, fig_title, fig_methods = next(f for f in FIGURES if f[0] == f"{name}.png")
                if sum(m in groups for m in fig_methods) >= 2:
                    figure_means(env, groups, fig_methods, fig_title, RESULTS / env / file)
                    section.append(f"![{title}: {fig_title}](results/{env}/{file})\n")
                if name == "main" and any(m in groups for m in methods):
                    figure_seeds(env, groups, methods, RESULTS / env / "main_seeds.png")
                    section.append(f"![{title}: every evaluation seed](results/{env}/main_seeds.png)\n")
        else:
            section.append("Evaluation runs are pending.\n")
        section.append(f"### Search\n\nFinal return of every search run.\n\n{search_table(env, runs, all_methods)}\n")
        parts.append("\n".join(section))

    head = ["Method"] + [ENV_INFO[e][0].split(",")[0].split(" (")[0] + (" + penalty" if e.endswith("penalty") else "")
                         for e in ENV_ORDER]
    lines = ["## Summary\n", "Final return in the evaluation runs: mean over seeds (lowest..highest seed).\n",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in [m for g in bench.GROUPS.values() for m in g]:
        row = [NAMES[m]]
        for env in ENV_ORDER:
            g = summary.get(env, {}).get(m)
            d = ENV_INFO[env][4]
            if not g:
                row.append("pending")
                continue
            finals = [r["final_return"] for r in g]
            partial = f" [{len(g)} of {num_seeds} seeds]" if len(g) < num_seeds else ""
            row.append(f"{fmt(np.mean(finals), d)} ({fmt(min(finals), d)}..{fmt(max(finals), d)}){partial}")
        lines.append("| " + " | ".join(row) + " |")
    parts.insert(1, "\n".join(lines) + "\n")
    parts.insert(2, runs_section(all_runs))
    parts.insert(3, edge_section())
    parts.insert(4, LIMITATIONS)
    (ROOT / "REPORT.md").write_text("\n".join(parts))
    print("wrote REPORT.md and results/ for", ", ".join(summary))
    import dashboard  # imported here because dashboard.py imports this module

    dashboard.main()


if __name__ == "__main__":
    main()
