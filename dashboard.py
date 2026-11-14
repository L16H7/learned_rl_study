"""Write results/dashboard.html: the learning curves as interactive charts.

  uv run python dashboard.py     # report.py also runs this

Reads results/<env>/runs.csv and curves.csv (made by report.py) and writes one HTML page with the data
inside it. Open the page in a browser: hover to read values, drag to zoom, click a legend entry to
hide a method. The page loads the Plotly library from its CDN, so it needs an internet connection.
"""
import csv
import json

import bench
import report

PLOTLY = "https://cdn.plot.ly/plotly-2.35.2.min.js"
# The colours of report.py, and the same hues stepped for a dark background.
DARK = {report.BLUE: "#3987e5", report.ORANGE: "#d95926", report.AQUA: "#199e70", report.YELLOW: "#c98500",
        report.MAGENTA: "#d55181", report.GREEN: "#008300"}
DASHES = {"solid": "solid", report.DASH: "dash", report.DOT: "dot", report.DASHDOT: "dashdot"}
VIEWS = [("Main methods", bench.GROUPS["main"]),
         ("Disco103 at replay ratio 32 and 4", report.FIGURES[1][2]),
         ("Baselines with 64×64 and 512×512 networks", report.FIGURES[2][2]),
         ("All methods", [m for g in bench.GROUPS.values() for m in g])]


def env_data(env):
    """Everything the page shows for one environment, or None if it has no results yet."""
    folder = report.RESULTS / env
    if not (folder / "curves.csv").exists():
        return None
    title, success, _, _, decimals = report.ENV_INFO[env]
    steps = bench.ENVS[env][2]
    runs = {}
    for row in csv.DictReader(open(folder / "runs.csv")):
        key = (row["phase"], row["method"], row["setting"], row["seed"])
        runs[key] = dict(phase=row["phase"], method=row["method"], setting=row["setting"], seed=int(row["seed"]),
                         final=float(row["final_return"]), ret=[None] * report.POINTS, len=[None] * report.POINTS)
    for row in csv.DictReader(open(folder / "curves.csv")):
        run = runs[(row["phase"], row["method"], row["setting"], row["seed"])]
        i = round(int(row["env_steps"]) * report.POINTS / steps) - 1
        run["ret"][i] = float(row["mean_return_last100"])
        run["len"][i] = float(row["mean_length_last100"])
    return dict(key=env, title=title, steps=steps, success=success, decimals=decimals, runs=list(runs.values()))


def main():
    methods = {m: dict(name=report.NAMES[m], light=report.STYLE[m][0], dark=DARK[report.STYLE[m][0]],
                       dash=DASHES[report.STYLE[m][1]], weights=bench.METHODS[m].flag == "--weights")
               for m in VIEWS[-1][1]}
    envs = [data for data in map(env_data, report.ENV_ORDER) if data]
    data = dict(envs=envs, methods=methods, views=[dict(title=t, methods=m) for t, m in VIEWS],
                points=report.POINTS, window=report.WINDOW)
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    out = report.RESULTS / "dashboard.html"
    out.write_text(PAGE.replace("__PLOTLY__", PLOTLY).replace("__DATA__", text))
    print(f"wrote {out.relative_to(report.ROOT)} for", ", ".join(e["key"] for e in envs))


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>learned_rl_study: learning curves</title>
<script src="__PLOTLY__"></script>
<style>
  :root {
    --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --axis: #c3c2b7; --active: #0b0b0b; --active-ink: #ffffff;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --axis: #383835; --active: #f0efec; --active-ink: #0b0b0b;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--page); color: var(--ink);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1500px; margin: 0 auto; padding: 20px 16px 48px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  p.note { color: var(--ink2); margin: 0 0 14px; max-width: 110ch; }
  .row { display: flex; flex-wrap: wrap; gap: 8px 18px; align-items: center; margin-bottom: 10px; }
  .tabs { display: flex; flex-wrap: wrap; gap: 6px; }
  button.tab { font: inherit; color: var(--ink2); background: var(--surface); border: 1px solid var(--axis);
               border-radius: 6px; padding: 5px 11px; cursor: pointer; }
  button.tab[aria-pressed="true"] { background: var(--active); color: var(--active-ink); border-color: var(--active); }
  label { color: var(--ink2); display: inline-flex; align-items: center; gap: 6px; }
  select, input[type=range] { font: inherit; accent-color: var(--ink2); }
  select { color: var(--ink); background: var(--surface); border: 1px solid var(--axis); border-radius: 6px;
           padding: 4px 6px; }
  .card { background: var(--surface); border: 1px solid var(--grid); border-radius: 10px; padding: 12px; }
  #chart { width: 100%; height: clamp(440px, 68vh, 760px); }
  h2 { font-size: 15px; margin: 22px 0 8px; }
  .scroll { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
  th { color: var(--ink2); font-weight: 600; }
  td.num, th.num { text-align: right; }
  td.seeds { color: var(--ink2); white-space: normal; min-width: 260px; }
  .swatch { display: inline-block; width: 22px; height: 0; border-top: 3px solid; margin-right: 8px;
            vertical-align: middle; }
  .pending { color: var(--ink2); margin-top: 10px; }
</style>
</head>
<body>
<main>
  <h1>learned_rl_study: learning curves</h1>
  <p class="note" id="note"></p>
  <div class="row"><div class="tabs" id="envs"></div></div>
  <div class="row">
    <div class="tabs" id="phases"></div>
    <div class="tabs" id="metrics"></div>
    <label>Methods <select id="view"></select></label>
    <label><input type="checkbox" id="seeds"> every seed</label>
    <label><input type="checkbox" id="band" checked> band from lowest to highest seed</label>
    <label>Smoothing <input type="range" id="smooth" min="0" max="0.95" step="0.05" value="0">
      <span id="smoothValue">0</span></label>
    <label>Hover <select id="hover">
      <option value="x unified">all methods at one step</option>
      <option value="closest">nearest curve</option>
    </select></label>
  </div>
  <div class="card"><div id="chart"></div></div>
  <h2 id="tableTitle"></h2>
  <div class="card scroll"><table id="table"></table></div>
  <div class="pending" id="pending"></div>
</main>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const PHASES = [["evaluation", "Evaluation runs"], ["search", "Search runs"]];
const METRICS = [["ret", "Return"], ["len", "Episode length"]];
const state = {env: D.envs[0].key, phase: "evaluation", metric: "ret", view: 0, seeds: false, band: true,
               smooth: 0, hover: "x unified"};

function tabs(id, items, field) {
  const box = $(id);
  for (const [value, text] of items) {
    const b = document.createElement("button");
    b.className = "tab"; b.textContent = text; b.dataset.value = value;
    b.onclick = () => {
      state[field] = value;
      if (field === "phase") { state.hover = value === "search" ? "closest" : "x unified"; $("hover").value = state.hover; }
      draw();
    };
    box.appendChild(b);
  }
}

function theme() {
  const s = getComputedStyle(document.documentElement), v = n => s.getPropertyValue(n).trim();
  return {surface: v("--surface"), ink: v("--ink"), ink2: v("--ink2"), muted: v("--muted"), grid: v("--grid"),
          axis: v("--axis"), dark: matchMedia("(prefers-color-scheme: dark)").matches};
}

function rgba(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${alpha})`;
}

// Exponential moving average with the start-up bias removed; missing points stay missing.
function smooth(values, weight) {
  if (!weight) return values;
  let last = 0, n = 0;
  return values.map(v => {
    if (v === null) return null;
    last = last * weight + (1 - weight) * v; n += 1;
    return last / (1 - Math.pow(weight, n));
  });
}

// Mean, lowest and highest value over runs at every grid point.
function across(runs, metric) {
  const mean = [], lo = [], hi = [];
  for (let i = 0; i < D.points; i++) {
    const vals = runs.map(r => r[metric][i]).filter(v => v !== null);
    mean.push(vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null);
    lo.push(vals.length ? Math.min(...vals) : null);
    hi.push(vals.length ? Math.max(...vals) : null);
  }
  return {mean, lo, hi};
}

function settingText(method, setting) {
  return D.methods[method].weights ? `${setting} weights` : `learning rate ${setting}`;
}

// method -> {setting -> runs} for one phase, and the setting each method was evaluated with.
function grouped(env, phase) {
  const by = {}, chosen = {};
  for (const r of env.runs) {
    if (r.phase === phase) ((by[r.method] ??= {})[r.setting] ??= []).push(r);
    if (r.phase === "evaluation") ((chosen[r.method] ??= {})[r.setting] ??= []).push(r);
  }
  for (const m in chosen) {
    chosen[m] = Object.keys(chosen[m]).sort((a, b) => chosen[m][b].length - chosen[m][a].length)[0];
  }
  for (const m in by) for (const s in by[m]) by[m][s].sort((a, b) => a.seed - b.seed);
  return {by, chosen};
}

// The curves to draw: one per method (evaluation) or one per method and setting (search).
function curves(env) {
  const {by, chosen} = grouped(env, state.phase), out = [], pending = [];
  for (const m of D.views[state.view].methods) {
    const settings = by[m] || {};
    const names = state.phase === "evaluation" ? (chosen[m] in settings ? [chosen[m]] : []) : Object.keys(settings);
    if (!names.length) { pending.push(D.methods[m].name); continue; }
    for (const s of names) {
      out.push({method: m, setting: s, runs: settings[s],
                main: state.phase === "evaluation" || s === chosen[m] || !(m in chosen)});
    }
  }
  return {list: out, pending};
}

function draw() {
  const env = D.envs.find(e => e.key === state.env), t = theme(), {list, pending} = curves(env);
  const x = Array.from({length: D.points}, (_, i) => (i + 1) * env.steps / D.points);
  const digits = state.metric === "ret" ? env.decimals + 1 : 1, f = `.${digits}f`;
  const unified = state.hover === "x unified", traces = [];
  for (const c of list) {
    const info = D.methods[c.method], color = t.dark ? info.dark : info.light;
    const name = state.phase === "evaluation" ? info.name : `${info.name}, ${settingText(c.method, c.setting)}`;
    const group = `${c.method}/${c.setting}`, a = across(c.runs, state.metric);
    const mean = smooth(a.mean, state.smooth), lo = smooth(a.lo, state.smooth), hi = smooth(a.hi, state.smooth);
    if (state.band && c.main) {
      const edge = {x, mode: "lines", line: {width: 0}, hoverinfo: "skip", showlegend: false, legendgroup: group};
      traces.push({...edge, y: lo}, {...edge, y: hi, fill: "tonexty", fillcolor: rgba(color, 0.14)});
    }
    if (state.seeds) {
      for (const r of c.runs) {
        traces.push({x, y: smooth(r[state.metric], state.smooth), mode: "lines", legendgroup: group, showlegend: false,
                     opacity: 0.45, line: {color, width: 1, dash: info.dash}, name: `${name}, seed ${r.seed}`,
                     ...(unified ? {hoverinfo: "skip"} : {hovertemplate: `%{y:${f}}<extra>${name}, seed ${r.seed}</extra>`})});
      }
    }
    traces.push({x, y: mean, mode: "lines", name, legendgroup: group, opacity: c.main ? 1 : 0.55,
                 line: {color, width: c.main ? 2.5 : 1.2, dash: info.dash},
                 customdata: lo.map((v, i) => [v, hi[i], c.runs.length]),
                 hovertemplate: `%{y:${f}}  (lowest %{customdata[0]:${f}}, highest %{customdata[1]:${f}}, ` +
                                `%{customdata[2]} seeds)<extra>${name}</extra>`});
  }
  const threshold = state.metric === "ret" && env.success !== null;
  const wide = window.innerWidth >= 1000;
  const layout = {
    paper_bgcolor: t.surface, plot_bgcolor: t.surface, hovermode: state.hover,
    font: {family: 'system-ui, -apple-system, "Segoe UI", sans-serif', size: 13, color: t.ink2},
    margin: {l: 66, r: 20, t: 14, b: 50},
    xaxis: {title: {text: "environment steps"}, range: [0, env.steps], tickformat: "~s", gridcolor: t.grid,
            linecolor: t.axis, zeroline: false, hoverformat: ",.0f"},
    yaxis: {title: {text: state.metric === "ret" ? "return, mean of the last 100 episodes"
                                                 : "episode length, mean of the last 100 episodes"},
            gridcolor: t.grid, linecolor: t.axis, zeroline: false},
    legend: wide ? {font: {color: t.ink}, tracegroupgap: 2} : {orientation: "h", y: -0.2, font: {color: t.ink}},
    hoverlabel: {bgcolor: t.surface, bordercolor: t.axis, font: {color: t.ink, size: 13}},
    shapes: threshold ? [{type: "line", xref: "paper", x0: 0, x1: 1, y0: env.success, y1: env.success,
                          line: {color: t.muted, width: 1, dash: "dash"}}] : [],
    annotations: threshold ? [{xref: "paper", x: 1, xanchor: "right", y: env.success, yanchor: "top",
                               text: `success threshold ${env.success}`, showarrow: false,
                               font: {color: t.muted, size: 12}}] : [],
    uirevision: [state.env, state.phase, state.metric, state.view].join("/"),
  };
  Plotly.react("chart", traces, layout, {responsive: true, displaylogo: false,
                                         modeBarButtonsToRemove: ["lasso2d", "select2d"]});

  for (const [id, field] of [["envs", "env"], ["phases", "phase"], ["metrics", "metric"]]) {
    for (const b of $(id).children) b.setAttribute("aria-pressed", String(b.dataset.value === String(state[field])));
  }
  $("note").textContent = `${env.title}: ${env.steps.toLocaleString("en")} environment steps per run. ` +
    `Every curve is a trailing mean over the last ${D.window} episodes. ` +
    (state.phase === "evaluation"
      ? "Evaluation runs: the chosen setting of each method with new seeds; the line is the mean over seeds. "
      : "Search runs: every setting that was tried; the line is the mean over the search seeds, and the setting " +
        "that was chosen is drawn bold. ") +
    "Drag to zoom, double-click to reset, click a legend entry to hide it.";
  table(env, list, t);
  $("pending").textContent = pending.length ? `No finished ${state.phase} runs yet: ${pending.join("; ")}.` : "";
}

function table(env, list, t) {
  const d = env.decimals, num = v => v.toFixed(d).replace("-", "−");
  const rows = list.map(c => {
    const finals = c.runs.map(r => r.final);
    return {c, finals, mean: finals.reduce((a, b) => a + b, 0) / finals.length};
  }).sort((a, b) => b.mean - a.mean);
  $("tableTitle").textContent = `Final return (mean of the last ${D.window} episodes of a run), ` +
    (state.phase === "evaluation" ? "evaluation runs" : "search runs");
  const head = ["Method", "Setting", "Seeds", "Mean", "Lowest seed", "Highest seed", "Each seed"];
  const el = (tag, text, cls) => { const e = document.createElement(tag); e.textContent = text; if (cls) e.className = cls; return e; };
  const tbl = $("table");
  tbl.replaceChildren();
  const hr = tbl.insertRow();
  head.forEach((h, i) => hr.appendChild(el("th", h, i >= 2 && i <= 5 ? "num" : "")));
  for (const {c, finals, mean} of rows) {
    const info = D.methods[c.method], tr = tbl.insertRow();
    const name = el("td", "");
    const sw = el("span", "", "swatch");
    sw.style.borderTopColor = t.dark ? info.dark : info.light;
    sw.style.borderTopStyle = info.dash === "solid" ? "solid" : info.dash === "dot" ? "dotted" : "dashed";
    name.append(sw, info.name);
    tr.append(name, el("td", settingText(c.method, c.setting) + (state.phase === "search" && c.main ? " (chosen)" : "")),
              el("td", String(finals.length), "num"), el("td", num(mean), "num"),
              el("td", num(Math.min(...finals)), "num"), el("td", num(Math.max(...finals)), "num"),
              el("td", finals.map(num).join(", "), "seeds"));
  }
}

tabs("envs", D.envs.map(e => [e.key, e.title.split(" (")[0].split(",")[0]]), "env");
tabs("phases", PHASES, "phase");
tabs("metrics", METRICS, "metric");
D.views.forEach((v, i) => $("view").add(new Option(v.title, i)));
$("view").onchange = e => { state.view = Number(e.target.value); draw(); };
$("seeds").onchange = e => { state.seeds = e.target.checked; draw(); };
$("band").onchange = e => { state.band = e.target.checked; draw(); };
$("smooth").oninput = e => { state.smooth = Number(e.target.value); $("smoothValue").textContent = e.target.value; draw(); };
$("hover").onchange = e => { state.hover = e.target.value; draw(); };
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
window.addEventListener("resize", () => { clearTimeout(window.resizeTimer); window.resizeTimer = setTimeout(draw, 200); });
draw();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
