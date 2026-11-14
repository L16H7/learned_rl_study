"""Write results/dashboard.html: the learning curves as interactive charts.

  uv run python dashboard.py     # report.py also runs this

Reads results/<env>/runs.csv and curves.csv (made by report.py) and writes one HTML page with the data
inside it. Open the page in a browser. Hovering lists every curve at that step, highest value first;
drag to zoom or pan, scroll to zoom, double-click to reset, click a legend entry to hide a method.
The page loads the Plotly library from its CDN, so it needs an internet connection.
"""
import csv
import json

import bench
import report

PLOTLY = "https://cdn.plot.ly/plotly-2.35.2.min.js"
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
    methods = {m: dict(name=report.NAMES[m], color=report.STYLE[m][0],
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
  :root { color-scheme: light; --page: #f6f6f4; --surface: #ffffff; --ink: #111111; --ink2: #444444;
          --muted: #6b6b6b; --line: #dadada; --grid: #ececec; --axis: #9a9a9a; --accent: #111111; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--page); color: var(--ink);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1500px; margin: 0 auto; padding: 20px 16px 48px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  p.note { color: var(--ink2); margin: 0 0 14px; max-width: 115ch; }
  .row { display: flex; flex-wrap: wrap; gap: 8px 18px; align-items: center; margin-bottom: 10px; }
  .tabs { display: flex; flex-wrap: wrap; gap: 6px; }
  button.tab { font: inherit; color: var(--ink2); background: var(--surface); border: 1px solid var(--line);
               border-radius: 6px; padding: 5px 11px; cursor: pointer; }
  button.tab:hover { border-color: var(--axis); }
  button.tab[aria-pressed="true"] { background: var(--accent); color: #fff; border-color: var(--accent); }
  label { color: var(--ink2); display: inline-flex; align-items: center; gap: 6px; }
  select { font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--line);
           border-radius: 6px; padding: 4px 6px; }
  input[type=range], input[type=checkbox] { accent-color: var(--accent); }
  .card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 12px; }
  .chartwrap { position: relative; }
  #chart { width: 100%; height: clamp(440px, 68vh, 760px); }
  #tip { position: absolute; pointer-events: none; display: none; z-index: 10; background: #fff;
         border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,.12);
         padding: 8px 10px; font-size: 12.5px; min-width: 260px; max-width: 460px; }
  #tip .head { color: var(--muted); margin-bottom: 4px; font-variant-numeric: tabular-nums; }
  #tip table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  #tip td { padding: 2px 4px; border: 0; white-space: nowrap; }
  #tip td.v { text-align: right; font-weight: 600; color: var(--ink); }
  #tip td.r { text-align: right; color: var(--muted); }
  #tip tr.near td { background: #f1f1ef; }
  #tip td.n { color: var(--ink); overflow: hidden; text-overflow: ellipsis; max-width: 240px; }
  #vline { position: absolute; width: 1px; background: #8a8a8a; pointer-events: none; display: none; z-index: 5; }
  h2 { font-size: 15px; margin: 22px 0 8px; }
  .scroll { overflow-x: auto; }
  table.res { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  table.res th, table.res td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
  table.res th { color: var(--ink2); font-weight: 600; }
  td.num, th.num { text-align: right !important; }
  td.seeds { color: var(--ink2); white-space: normal !important; min-width: 260px; }
  .swatch { display: inline-block; width: 22px; height: 0; border-top: 3px solid; margin-right: 8px; vertical-align: middle; }
  .pending { color: var(--ink2); margin-top: 10px; }
  .hint { color: var(--muted); font-size: 12.5px; }
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
  </div>
  <div class="row">
    <div class="tabs" id="drag"></div>
    <button class="tab" id="reset">Reset view</button>
    <span class="hint">Scroll over the chart to zoom, double-click to reset, click a legend entry to hide a curve.</span>
  </div>
  <div class="card"><div class="chartwrap"><div id="chart"></div><div id="vline"></div><div id="tip"></div></div></div>
  <h2 id="tableTitle"></h2>
  <div class="card scroll"><table class="res" id="table"></table></div>
  <div class="pending" id="pending"></div>
</main>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const chart = $("chart");
const PHASES = [["evaluation", "Evaluation runs"], ["search", "Search runs"]];
const METRICS = [["ret", "Return"], ["len", "Episode length"]];
const DRAGS = [["zoom", "Box zoom"], ["zoomx", "Zoom steps only"], ["pan", "Pan"]];
const T = {surface: "#ffffff", ink: "#111111", ink2: "#444444", muted: "#6b6b6b", grid: "#ececec", axis: "#9a9a9a"};
const state = {env: D.envs[0].key, phase: "evaluation", metric: "ret", view: 0, seeds: false, band: true,
               smooth: 0, drag: "zoom"};
let current = null;  // the environment and number format of the drawn chart

function tabs(id, items, field) {
  for (const [value, text] of items) {
    const b = document.createElement("button");
    b.className = "tab"; b.textContent = text; b.dataset.value = value;
    b.onclick = () => { state[field] = value; draw(); };
    $(id).appendChild(b);
  }
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
  hideTip();
  const env = D.envs.find(e => e.key === state.env), {list, pending} = curves(env);
  const x = Array.from({length: D.points}, (_, i) => (i + 1) * env.steps / D.points);
  const digits = state.metric === "ret" ? env.decimals + 1 : 1;
  current = {env, digits};
  const traces = [];
  for (const c of list) {
    const info = D.methods[c.method], color = info.color;
    const name = state.phase === "evaluation" ? info.name : `${info.name}, ${settingText(c.method, c.setting)}`;
    const group = `${c.method}/${c.setting}`, a = across(c.runs, state.metric);
    const mean = smooth(a.mean, state.smooth), lo = smooth(a.lo, state.smooth), hi = smooth(a.hi, state.smooth);
    if (state.band && c.main) {
      const edge = {x, mode: "lines", line: {width: 0}, hoverinfo: "skip", showlegend: false, legendgroup: group};
      traces.push({...edge, y: lo}, {...edge, y: hi, fill: "tonexty", fillcolor: rgba(color, 0.09)});
    }
    if (state.seeds) {
      for (const r of c.runs) {
        traces.push({x, y: smooth(r[state.metric], state.smooth), mode: "lines", legendgroup: group, showlegend: false,
                     opacity: 0.4, line: {color, width: 1, dash: info.dash}, hoverinfo: "skip"});
      }
    }
    traces.push({x, y: mean, mode: "lines", name, legendgroup: group, opacity: c.main ? 1 : 0.5,
                 line: {color, width: c.main ? 2.4 : 1.3, dash: info.dash}, hoverinfo: "none",
                 meta: {mean: true, name, color, dash: info.dash, lo, hi, n: c.runs.length, main: c.main}});
  }
  const threshold = state.metric === "ret" && env.success !== null;
  const wide = window.innerWidth >= 1000;
  const dragmode = state.drag === "zoomx" ? "zoom" : state.drag;
  const layout = {
    paper_bgcolor: T.surface, plot_bgcolor: T.surface, hovermode: "x", dragmode,
    font: {family: 'system-ui, -apple-system, "Segoe UI", sans-serif', size: 13, color: T.ink2},
    margin: {l: 70, r: 20, t: 14, b: 52},
    xaxis: {title: {text: "environment steps"}, range: [0, env.steps], tickformat: "~s", gridcolor: T.grid,
            linecolor: T.axis, zeroline: false, showline: true, ticks: "outside", tickcolor: T.axis},
    yaxis: {title: {text: state.metric === "ret" ? "return, mean of the last 100 episodes"
                                                 : "episode length, mean of the last 100 episodes"},
            gridcolor: T.grid, linecolor: T.axis, zeroline: false, showline: true, ticks: "outside",
            tickcolor: T.axis, fixedrange: state.drag === "zoomx"},
    legend: wide ? {font: {color: T.ink}, tracegroupgap: 2, bgcolor: "rgba(255,255,255,0.85)"}
                 : {orientation: "h", y: -0.2, font: {color: T.ink}},
    shapes: threshold ? [{type: "line", xref: "paper", x0: 0, x1: 1, y0: env.success, y1: env.success,
                          line: {color: T.muted, width: 1, dash: "dash"}}] : [],
    annotations: threshold ? [{xref: "paper", x: 1, xanchor: "right", y: env.success, yanchor: "top",
                               text: `success threshold ${env.success}`, showarrow: false,
                               font: {color: T.muted, size: 12}}] : [],
    uirevision: [state.env, state.phase, state.metric, state.view].join("/"),
  };
  Plotly.react(chart, traces, layout, {responsive: true, displaylogo: false, scrollZoom: true,
                                       modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"]});

  for (const [id, field] of [["envs", "env"], ["phases", "phase"], ["metrics", "metric"], ["drag", "drag"]]) {
    for (const b of $(id).children) b.setAttribute("aria-pressed", String(b.dataset.value === String(state[field])));
  }
  $("note").textContent = `${env.title}: ${env.steps.toLocaleString("en")} environment steps per run. ` +
    `Every curve is a trailing mean over the last ${D.window} episodes. ` +
    (state.phase === "evaluation"
      ? "Evaluation runs: the chosen setting of each method with new seeds; the line is the mean over seeds. "
      : "Search runs: every setting that was tried; the line is the mean over the search seeds, and the chosen " +
        "setting is drawn bold. ") +
    "Hover to list every curve at that step, highest first.";
  table(env, list);
  $("pending").textContent = pending.length ? `No finished ${state.phase} runs yet: ${pending.join("; ")}.` : "";
}

function table(env, list) {
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
    const name = el("td", ""), sw = el("span", "", "swatch");
    sw.style.borderTopColor = info.color;
    sw.style.borderTopStyle = info.dash === "solid" ? "solid" : info.dash === "dot" ? "dotted" : "dashed";
    name.append(sw, info.name);
    tr.append(name, el("td", settingText(c.method, c.setting) + (state.phase === "search" && c.main ? " (chosen)" : "")),
              el("td", String(finals.length), "num"), el("td", num(mean), "num"),
              el("td", num(Math.min(...finals)), "num"), el("td", num(Math.max(...finals)), "num"),
              el("td", finals.map(num).join(", "), "seeds"));
  }
}

function resetView() {
  if (!current) return;
  Plotly.relayout(chart, {"xaxis.range": [0, current.env.steps], "yaxis.autorange": true});
}

// --- hover box: every visible curve at the hovered step, sorted by value ---
const tip = $("tip"), vline = $("vline");
function hideTip() { tip.style.display = "none"; vline.style.display = "none"; }
const fmt = (v, d) => (v === null || v === undefined || isNaN(v)) ? "–" : v.toFixed(d).replace("-", "−");
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

chart.addEventListener("mousemove", ev => { chart._mouse = ev; });
function onHover(ev) {
  if (!current) return;
  const pts = ev.points.filter(p => p.data.meta && p.data.meta.mean && p.y !== null && p.y !== undefined);
  if (!pts.length) { hideTip(); return; }
  const full = chart._fullLayout, xa = full.xaxis, ya = full.yaxis;
  const box = chart.getBoundingClientRect(), mouse = ev.event || chart._mouse;
  const step = pts[0].x, i = pts[0].pointIndex, d = current.digits;
  const mx = mouse ? mouse.clientX - box.left : xa._offset + xa.l2p(step);
  const my = mouse ? mouse.clientY - box.top : ya._offset + ya._length / 2;
  const cursorY = ya.p2l(my - ya._offset);
  const rows = pts.map(p => ({m: p.data.meta, y: p.y, lo: p.data.meta.lo[i], hi: p.data.meta.hi[i]}))
                  .sort((a, b) => b.y - a.y);
  let near = 0;
  rows.forEach((r, k) => { if (Math.abs(r.y - cursorY) < Math.abs(rows[near].y - cursorY)) near = k; });
  const dash = s => s === "solid" ? "solid" : s === "dot" ? "dotted" : "dashed";
  tip.innerHTML = `<div class="head">step ${Math.round(step).toLocaleString("en")}</div><table>` +
    rows.map((r, k) => `<tr class="${k === near ? "near" : ""}"><td><span class="swatch" style="border-top-color:${r.m.color};` +
      `border-top-style:${dash(r.m.dash)}"></span></td><td class="n">${esc(r.m.name)}</td>` +
      `<td class="v">${fmt(r.y, d)}</td><td class="r">${fmt(r.lo, d)} .. ${fmt(r.hi, d)}</td></tr>`).join("") +
    `</table><div class="head" style="margin:4px 0 0">value, then lowest .. highest seed (${rows[0].m.n} seeds)</div>`;
  tip.style.display = "block";
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let left = mx + 18; if (left + w > box.width - 4) left = mx - w - 18;
  let top = my - h / 2; top = Math.max(4, Math.min(top, box.height - h - 4));
  tip.style.left = `${Math.max(4, left)}px`; tip.style.top = `${top}px`;
  vline.style.display = "block";
  vline.style.left = `${xa._offset + xa.l2p(step)}px`; vline.style.top = `${ya._offset}px`;
  vline.style.height = `${ya._length}px`;
}

tabs("envs", D.envs.map(e => [e.key, e.title.split(" (")[0].split(",")[0]]), "env");
tabs("phases", PHASES, "phase");
tabs("metrics", METRICS, "metric");
tabs("drag", DRAGS, "drag");
D.views.forEach((v, i) => $("view").add(new Option(v.title, i)));
$("view").onchange = e => { state.view = Number(e.target.value); draw(); };
$("seeds").onchange = e => { state.seeds = e.target.checked; draw(); };
$("band").onchange = e => { state.band = e.target.checked; draw(); };
$("smooth").oninput = e => { state.smooth = Number(e.target.value); $("smoothValue").textContent = e.target.value; draw(); };
$("reset").onclick = resetView;
draw();
chart.on("plotly_hover", onHover);
chart.on("plotly_unhover", hideTip);
chart.on("plotly_relayout", e => {
  hideTip();
  // "Zoom steps only": fit the return axis to the curves inside the chosen step range
  if (state.drag !== "zoomx" || !("xaxis.range[0]" in e || "xaxis.range" in e)) return;
  const [x0, x1] = chart._fullLayout.xaxis.range;
  let lo = Infinity, hi = -Infinity;
  for (const t of chart.data) {
    if (t.visible === "legendonly" || (t.line && t.line.width === 0 && !t.fill && !state.band)) continue;
    t.x.forEach((x, i) => { const y = t.y[i]; if (x >= x0 && x <= x1 && y !== null) { lo = Math.min(lo, y); hi = Math.max(hi, y); } });
  }
  if (lo < hi) { const pad = 0.05 * (hi - lo); Plotly.relayout(chart, {"yaxis.range": [lo - pad, hi + pad]}); }
});
chart.on("plotly_doubleclick", () => { resetView(); return false; });
chart.addEventListener("mouseleave", hideTip);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
