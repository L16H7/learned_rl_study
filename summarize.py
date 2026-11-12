"""Rank the configurations in a folder of runs by final return.

Runs that differ only in seed are grouped. Only the settings that vary across the folder are shown.
Unfinished runs are left out.

  uv run python summarize.py runs/a2c_tune --top 15
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

IGNORED = {"out", "seed"}
DEFAULTS = {"optimizer": "adam", "normalize_adv": False}  # options added after some runs were made


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+", help="folders holding one sub-folder per run")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--rule", default=None, help="only runs of this method, e.g. a2c")
    args = p.parse_args()

    groups, unfinished = defaultdict(list), 0
    for folder in args.folders:
        for d in sorted(Path(folder).iterdir()):
            if not (d / "episodes.csv").exists():
                continue
            cfg = json.loads((d / "args.json").read_text())
            if args.rule and cfg["rule"] != args.rule:
                continue
            rows = list(csv.DictReader(open(d / "episodes.csv")))
            if len(rows) < 100 or int(rows[-1]["env_steps"]) < 0.98 * cfg["env_steps"]:
                unfinished += 1
                continue
            ret = np.array([float(r["return"]) for r in rows[-100:]])
            if cfg["rule"] == "a2c":
                cfg = {**DEFAULTS, **cfg}
            key = tuple(sorted((k, json.dumps(v)) for k, v in cfg.items() if k not in IGNORED))
            groups[key].append((cfg["seed"], ret.mean(), np.mean(ret >= 200)))

    varying = sorted({k for key in groups for k, _ in key
                      if len({dict(other).get(k) for other in groups}) > 1})
    ranked = sorted(groups.items(), key=lambda kv: -np.mean([r[1] for r in kv[1]]))
    print(f"{len(groups)} configurations, {sum(len(v) for v in groups.values())} finished runs"
          + (f", {unfinished} unfinished runs left out" if unfinished else ""))
    print(f"{'final return':>12s} {'min..max':>12s} {'landed':>7s} {'seeds':>5s}  settings that vary")
    for key, runs in ranked[:args.top]:
        finals = [r[1] for r in runs]
        settings = ", ".join(f"{k}={dict(key).get(k, '-')}" for k in varying)
        print(f"{np.mean(finals):12.0f} {min(finals):5.0f}..{max(finals):<5.0f} {np.mean([r[2] for r in runs]):7.0%} "
              f"{len(runs):5d}  {settings}")
    print("final return = mean of each run's last 100 episodes, averaged over seeds")


if __name__ == "__main__":
    main()
