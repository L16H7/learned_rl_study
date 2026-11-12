"""Compare training runs: return, episode length and success rate against environment steps.

Runs are grouped by method, agent width and learning rate (for OPEN, which has no learning rate:
by the weights used); seeds within a group are averaged. For each method the best group by final
return is drawn bold with a band from the worst to the best seed; its other groups are thin and
faded. One figure is written per agent width (the first hidden size of the agent's network), or a
single figure with --one_figure, where a method's wider agent is drawn dashed. Disco103 with replay
ratio 4 is a separate method from Disco103 with the reference ratio 32 and is drawn dotted in the same
colour. Unfinished runs are left out. A summary table is printed.

  uv run python plot.py runs/cartpole_tune              # writes runs/cartpole_tune/comparison_w64.png, ...
  uv run python plot.py runs/cartpole_final --one_figure
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = {  # fixed order and colour per method
    "disco": ("Disco103", "#2a78d6"),
    "disco_rr4": ("Disco103, replay ratio 4", "#2a78d6"),
    "a2c": ("A2C", "#eb6834"),
    "ppo": ("PPO", "#1baf7a"),
    "reinforce": ("REINFORCE", "#eda100"),
    "open": ("PPO + OPEN", "#e87ba4"),
    "ppo_ref": ("PPO + Adam, OPEN's loop", "#008300"),
    "actor_critic": ("disco_rl actor-critic", "#6250d6"),
}
# Per environment: the return that counts as success, what success is called, the return-axis
# limits, and whether episode length says anything the return does not.
ENVS = {
    "LunarLander-v3": dict(success=200, word="landed", ylim=(-400, 320), length_panel=True),
    "CartPole-v1": dict(success=475, word="solved", ylim=(0, 520), length_panel=False),
    "Acrobot-v1": dict(success=-100, word="solved", ylim=(-520, 0), length_panel=False),
    "MinAtar/Breakout-v1": dict(success=None, word="", ylim=None, length_panel=True),
}
UNKNOWN_ENV = dict(success=None, word="", ylim=None, length_panel=True)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
WINDOW = 50  # episodes in every rolling mean


def style(rule):
    """(display name, colour, line style, sort index) for a method key such as "ppo" or "disco@512+"."""
    base, _, width = rule.partition("@")
    name, color = METHODS[base]
    wider = width.endswith("+")  # the wider of two agent sizes of the same method
    width = width.rstrip("+")
    if base == "disco_rr4":
        linestyle = (0, (4, 2, 1, 2)) if wider else (0, (1, 2))
    else:
        linestyle = (0, (4, 2)) if wider else "solid"
    return (f"{name} ({width}-wide)" if width else name), color, linestyle, (list(METHODS).index(base), wider)


def variant_text(rule, variant):
    return f"{variant} weights" if rule.startswith("open") else f"learning rate {variant:g}"


def rolling(x, w):
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    return (c[w:] - c[:-w]) / w


def load(folder: Path):
    """One dict per finished run with per-episode arrays. Returns (runs, names of unfinished runs)."""
    runs, unfinished = [], []
    for d in sorted(folder.iterdir()):
        if not (d / "episodes.csv").exists():
            continue
        cfg = json.loads((d / "args.json").read_text())
        rows = list(csv.DictReader(open(d / "episodes.csv")))
        if len(rows) < 2 * WINDOW or int(rows[-1]["env_steps"]) < 0.98 * cfg["env_steps"]:
            unfinished.append(d.name)
            continue
        env = cfg.get("env", "LunarLander-v3")
        success = ENVS.get(env, UNKNOWN_ENV)["success"]
        ret = np.array([float(r["return"]) for r in rows])
        steps = np.array([int(r["env_steps"]) for r in rows])
        solved = np.flatnonzero(rolling(ret, 100) >= success) if success is not None else []
        width = (cfg.get("torso") or cfg.get("hidden") or [512, 512])[0]  # disco_rl agents call it torso
        rule = cfg["rule"]
        if rule == "disco" and cfg.get("replay_ratio", 4) != 32:  # runs made before the option existed used 4
            rule = "disco_rr4"
        runs.append(dict(
            name=d.name, env=env, rule=rule, width=width, seed=cfg["seed"], steps=steps, ret=ret,
            variant=cfg["weights"] if cfg["rule"] == "open" else cfg["lr"],
            length=np.array([int(r["length"]) for r in rows]), final=ret[-100:].mean(),
            success_rate=np.mean(ret[-100:] >= success) if success is not None else float("nan"),
            solved_at=steps[solved[0] + 99] if len(solved) else None))
    return runs, unfinished


def label_ends(ax, ends, gap=13.0):
    """Name each bold line at its right end. Labels that would overlap are spread vertically and
    joined to their line by a thin leader."""
    ends = sorted(ends, key=lambda e: e[1])
    ys = [ax.transData.transform((x, y))[1] for x, y, _ in ends]  # pixels
    placed = list(ys)
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    shift = (sum(ys) - sum(placed)) / max(len(ys), 1)  # keep the group centred on the lines
    scale = 72.0 / ax.figure.dpi
    for (x, y, name), y_px, y_label in zip(ends, ys, placed):
        dy = (y_label + shift - y_px) * scale
        moved = abs(dy) > 1.0
        ax.annotate(name, (x, y), xytext=(14 if moved else 6, dy), textcoords="offset points", va="center",
                    fontsize=9, color=INK, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.6, shrinkA=0, shrinkB=2) if moved else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="folder holding one sub-folder per run")
    p.add_argument("--title", default=None)
    p.add_argument("--one_figure", action="store_true", help="draw every agent width in one figure")
    args = p.parse_args()
    folder = Path(args.folder)
    all_runs, unfinished = load(folder)
    envs = sorted({r["env"] for r in all_runs})
    if len(envs) != 1:
        raise SystemExit(f"expected runs of one environment, found {envs}")
    title = args.title or f"{envs[0]}: {folder.name}"
    if args.one_figure:
        widths = defaultdict(set)
        for r in all_runs:
            widths[r["rule"]].add(r["width"])
        for r in all_runs:  # methods run at two widths become two lines: "rule@64" and "rule@512+"
            if len(widths[r["rule"]]) > 1:
                r["rule"] = f"{r['rule']}@{r['width']}" + ("+" if r["width"] == max(widths[r["rule"]]) else "")
        figure(all_runs, envs[0], title, folder / "comparison.png")
    else:
        for width in sorted({r["width"] for r in all_runs}):
            runs = [r for r in all_runs if r["width"] == width]
            figure(runs, envs[0], f"{title} — agent width {width}", folder / f"comparison_w{width}.png")
    if unfinished:
        print(f"left out {len(unfinished)} unfinished runs: {', '.join(unfinished)}")


def figure(runs, env, title, path):
    spec = ENVS.get(env, UNKNOWN_ENV)
    success, word = spec["success"], spec["word"]
    groups = defaultdict(list)
    for r in runs:
        groups[r["rule"], r["variant"]].append(r)
    order = sorted(groups, key=lambda k: (style(k[0])[3], str(k[1])))
    best = {}  # method -> variant with the highest mean final return
    for rule, variant in order:
        score = np.mean([r["final"] for r in groups[rule, variant]])
        if rule not in best or score > best[rule][1]:
            best[rule] = (variant, score)

    xmax = max(r["steps"][-1] for r in runs)
    grid = np.linspace(0, xmax, 400)
    panels = [("return", f"Episode return (mean of the last {WINDOW} episodes)", lambda r: rolling(r["ret"], WINDOW))]
    if spec["length_panel"]:
        panels.append(("length", f"Episode length in steps (mean of the last {WINDOW} episodes)",
                       lambda r: rolling(r["length"], WINDOW)))
    if success is not None:
        panels.append(("rate", f"Share of episodes with return of at least {success} (last {WINDOW} episodes)",
                       lambda r: rolling(r["ret"] >= success, WINDOW)))

    def on_grid(r, series):  # a run's rolling series on the common step grid; NaN outside its range
        x = r["steps"][WINDOW - 1:]
        return np.interp(grid, x, series(r), left=np.nan, right=np.nan)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.2 + 2.8 * len(panels)), sharex=True,
                             facecolor=SURFACE, squeeze=False)
    axes = axes[:, 0]
    for ax, (kind, panel_title, series) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.set_title(panel_title, loc="left", fontsize=11, color=INK, pad=8)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        ends = []  # (x, y, name) at the end of every bold line, for the direct labels
        for rule, variant in order:
            name, color, linestyle, _ = style(rule)
            curves = np.stack([on_grid(r, series) for r in groups[rule, variant]])
            covered = ~np.isnan(curves).any(axis=0)  # steps every seed has reached
            x, mean = grid[covered] / 1000, curves[:, covered].mean(axis=0)
            if best[rule][0] == variant:
                if len(curves) > 1:
                    ax.fill_between(x, curves[:, covered].min(axis=0), curves[:, covered].max(axis=0),
                                    color=color, alpha=0.15, linewidth=0, zorder=2)
                ax.plot(x, mean, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                        linestyle=linestyle, zorder=3,
                        label=f"{name}, {variant_text(rule, variant)} ({len(curves)} seeds)")
                ends.append((x[-1], mean[-1], name))
            else:
                ax.plot(x, mean, color=color, linewidth=1, alpha=0.4, zorder=2, linestyle=linestyle)
        ax.set_xlim(0, xmax / 1000 * 1.3)  # room on the right for the line labels
        ax.set_xticks([t for t in ax.get_xticks() if 0 <= t <= xmax / 1000 + 1])
        if kind == "return" and spec["ylim"]:
            ax.set_ylim(*spec["ylim"])
        elif kind == "rate":
            ax.set_ylim(-0.03, 1.03)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax._line_ends = ends
    if success is not None:
        axes[0].axhline(success, color=INK2, linewidth=1, zorder=1)
        axes[0].annotate(f"{success} = {word}", (0, success), xytext=(4, 4), textcoords="offset points",
                         fontsize=9, color=INK2)
    axes[-1].set_xlabel("Environment steps (thousands)")
    handles, labels = axes[0].get_legend_handles_labels()
    height = fig.get_figheight()
    legend_rows = (len(handles) + 1) // 2
    fig.suptitle(title, x=0.065, y=1 - 0.3 / height, ha="left", fontsize=13, color=INK)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.055, 1 - 0.42 / height), ncol=2, frameon=False,
               fontsize=9, handlelength=1.6, columnspacing=1.8)
    note_top = 1 - (0.62 + 0.24 * legend_rows) / height
    cut = f" Return axis cut at {spec['ylim'][0]:g}." if spec["ylim"] and spec["ylim"][0] < 0 else ""
    fig.text(0.065, note_top, "Bold line: mean over seeds for the method's best setting, with a band from the worst "
             f"to the best seed.\nThin lines: the method's other settings.{cut}", fontsize=8.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, note_top - 0.45 / height))
    fig.canvas.draw()  # fix the layout so that label positions can be computed in pixels
    for ax in axes:
        label_ends(ax, ax._line_ends)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)

    print(f"\n{title}  ->  {path}")
    reached = f"first reached {success}" if success is not None else ""
    print(f"{'method':38s} {'setting':>22s} {'seeds':>5s} {'final return (min..max)':>26s} {word:>7s}  {reached}")
    for rule, variant in order:
        g = groups[rule, variant]
        finals = [r["final"] for r in g]
        solved = ", ".join(f"{r['solved_at'] / 1000:.0f}k" if r["solved_at"] else "never" for r in g)
        mark = " *" if best[rule][0] == variant else ""
        rate = f"{np.mean([r['success_rate'] for r in g]):7.0%}" if success is not None else ""
        print(f"{style(rule)[0]:38s} {variant_text(rule, variant):>22s} {len(g):5d} "
              f"{np.mean(finals):9.0f} ({min(finals):5.0f}..{max(finals):5.0f}) {rate:>12s}  "
              f"{solved if success is not None else ''}{mark}")
    print("final return = mean of each run's last 100 episodes; * = best setting for the method")


if __name__ == "__main__":
    main()
