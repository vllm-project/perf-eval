#!/usr/bin/env python3
"""Generate throughput-vs-concurrency scaling reports from perf-eval results.

One HTML report per (model, ISL/OSL combination), with Chart.js line charts
across concurrency levels — one line per workload variant (e.g. mi355-ut,
vllm-ci). Workload variants that share the same model slug are overlaid on
the same chart set.

Usage:
    python3 gen_scaling_report.py [results_dir]

Output files are written next to this script:
    scaling-<model-slug>.html    one per model group
    scaling-index.html           landing page tabbing between models

Model grouping:
    Result directory names are split on the first '-' that separates a known
    prefix (mi355-ut, vllm-ci, upstream) from the model portion.  Directories
    whose names don't match any known prefix are treated as their own group
    with the full name as the label.

    Directories prefixed with attn-sweep- or moe-sweep- are excluded — those
    are handled by gen_report.py.

    Example:
        mi355-ut-gpt-oss-120b-mi355x  →  group "gpt-oss-120b-mi355x", label "mi355-ut"
        vllm-ci-gpt-oss-120b-mi355x   →  group "gpt-oss-120b-mi355x", label "vllm-ci"
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
OUT_DIR = Path(__file__).parent

# Directories starting with these prefixes belong to gen_report.py, not here.
EXCLUDED_PREFIXES = ("attn-sweep-", "moe-sweep-")

# Prefixes stripped to derive the model group key. Order matters: longer
# prefixes must come before any prefix that is a prefix of them.
KNOWN_PREFIXES = [
    "mi355-ut-",
    "vllm-ci-",
    "upstream-",
]

# ── JSON metric extraction ────────────────────────────────────────────────────

METRICS = [
    ("output_throughput",    "Output token throughput", "tok/s", True),
    ("request_throughput",   "Request throughput",      "req/s", True),
    ("total_token_throughput", "Total token throughput","tok/s", True),
    ("mean_ttft_ms",         "Mean TTFT",               "ms",    False),
    ("median_ttft_ms",       "Median TTFT",             "ms",    False),
    ("p99_ttft_ms",          "P99 TTFT",                "ms",    False),
    ("mean_tpot_ms",         "Mean TPOT",               "ms",    False),
    ("median_tpot_ms",       "Median TPOT",             "ms",    False),
    ("p99_tpot_ms",          "P99 TPOT",                "ms",    False),
    ("mean_itl_ms",          "Mean ITL",                "ms",    False),
    ("median_itl_ms",        "Median ITL",              "ms",    False),
    ("p99_itl_ms",           "P99 ITL",                 "ms",    False),
]

BENCH_NAME_RE = re.compile(
    r"bench-isl(?P<isl>\d+)-osl(?P<osl>\d+)-conc(?P<conc>\d+)\.json$"
)


def load_result_dir(result_dir: Path) -> dict:
    """Return {(isl, osl, conc): {metric_key: float}} for one result directory."""
    points = {}
    for jf in sorted(result_dir.glob("bench-isl*-osl*-conc*.json")):
        m = BENCH_NAME_RE.search(jf.name)
        if not m:
            continue
        isl, osl, conc = int(m["isl"]), int(m["osl"]), int(m["conc"])
        try:
            data = json.loads(jf.read_text())
        except Exception as e:
            print(f"  warning: could not parse {jf}: {e}", file=sys.stderr)
            continue
        points[(isl, osl, conc)] = {key: data.get(key) for key, *_ in METRICS}
    return points


# ── Directory grouping ────────────────────────────────────────────────────────

def split_prefix(dir_name: str) -> tuple[str, str]:
    """Return (label, model_group) for a result directory name."""
    for prefix in KNOWN_PREFIXES:
        if dir_name.startswith(prefix):
            return prefix.rstrip("-"), dir_name[len(prefix):]
    return dir_name, dir_name


def collect_groups(results_dir: Path) -> dict[str, dict[str, dict]]:
    """Return {model_group: {label: {(isl,osl,conc): metrics}}}."""
    groups: dict[str, dict] = defaultdict(dict)
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        if any(d.name.startswith(p) for p in EXCLUDED_PREFIXES):
            print(f"  skip {d.name} — handled by gen_report.py")
            continue
        label, group = split_prefix(d.name)
        points = load_result_dir(d)
        if not points:
            print(f"  skip {d.name} — no bench JSON files")
            continue
        # Multiple dirs with the same label in the same group (shouldn't
        # normally happen, but merge rather than overwrite)
        if label in groups[group]:
            groups[group][label].update(points)
        else:
            groups[group][label] = points
        print(f"  {d.name}  →  group={group!r}  label={label!r}  ({len(points)} points)")
    return dict(groups)


# ── HTML / JS generation ──────────────────────────────────────────────────────

# A palette that works for up to ~8 lines; first two match AFO-LLM's colours.
PALETTE = [
    "rgba(102, 126, 234, 1)",    # blue-violet  (mi355-ut / ATOM)
    "rgba(245, 158,  11, 1)",    # amber        (vllm-ci / ATOM-OOT)
    "rgba(244, 114, 182, 1)",    # pink         (upstream / vLLM)
    "rgba( 52, 211, 153, 1)",    # emerald      (MTP3)
    "rgba(251, 191,  36, 1)",    # yellow
    "rgba(167, 139, 250, 1)",    # violet
    "rgba( 34, 197,  94, 1)",    # green
    "rgba(249, 115,  22, 1)",    # orange
]

CSS = """\
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  color: #e0e0e0;
  min-height: 100vh;
  padding: 1.5rem 2rem 3rem;
}

header {
  background: rgba(255,255,255,.05);
  backdrop-filter: blur(10px);
  padding: 1.5rem 2rem;
  border-radius: 12px;
  margin-bottom: 1.75rem;
  box-shadow: 0 8px 32px rgba(0,0,0,.3);
}

h1 {
  font-size: 1.8rem;
  font-weight: 700;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: .3rem;
}

.subtitle { color: #94a3b8; font-size: .95rem; }

/* ISL/OSL tab bar */
.tab-bar {
  display: flex; gap: 0; margin-bottom: 1.5rem;
  border-bottom: 1px solid #1e293b;
}
.tab {
  padding: .5rem 1.4rem; font-size: .82rem; cursor: pointer;
  color: #64748b; border-bottom: 2px solid transparent;
  margin-bottom: -1px; user-select: none; transition: color .15s;
}
.tab:hover { color: #94a3b8; }
.tab.active { color: #38bdf8; border-bottom-color: #38bdf8; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }

/* KPI strip */
.kpi-strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem; margin-bottom: 1.5rem;
}
.kpi-card {
  background: rgba(255,255,255,.05);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 10px; padding: 1.1rem 1.4rem;
  transition: transform .2s, box-shadow .2s;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(102,126,234,.25); }
.kpi-label { font-size: .8rem; color: #94a3b8; margin-bottom: .4rem; }
.kpi-value { font-size: 1.8rem; font-weight: 700; }
.kpi-sub   { font-size: .75rem; color: #64748b; margin-top: .25rem; }
.pos { color: #4ade80; } .neg { color: #f87171; } .neu { color: #60a5fa; }

/* Chart grid */
.chart-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
  gap: 1.25rem; margin-bottom: 1.5rem;
}
.chart-card {
  background: rgba(255,255,255,.05);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 10px; padding: 1.25rem 1.5rem;
}
.chart-card h3 { font-size: 1rem; color: #cbd5e1; margin-bottom: 1rem; }

/* Detailed table */
.table-card {
  background: rgba(255,255,255,.05);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 10px; padding: 1.25rem 1.5rem;
  overflow-x: auto; margin-bottom: 1.5rem;
}
.table-card h3 { font-size: 1rem; color: #cbd5e1; margin-bottom: 1rem; }
table { width: 100%; border-collapse: collapse; }
th {
  padding: .6rem 1rem; text-align: left;
  background: rgba(255,255,255,.08); color: #667eea;
  font-size: .8rem; font-weight: 600; cursor: pointer; user-select: none;
}
th:hover { background: rgba(255,255,255,.12); }
td { padding: .55rem 1rem; border-bottom: 1px solid rgba(255,255,255,.05); font-size: .85rem; }
tr:hover td { background: rgba(255,255,255,.03); }
"""

INDEX_CSS = """\
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  color: #e0e0e0; font-size: 14px;
  display: flex; flex-direction: column;
}
.header { padding: .75rem 2rem 0; flex-shrink: 0; }
h1 { font-size: 1.2rem; font-weight: 600; color: #f8fafc; margin-bottom: .6rem; }
.model-tab-bar { display: flex; gap: 0; border-bottom: 1px solid #1e293b; }
.model-tab {
  padding: .5rem 1.4rem; font-size: .8rem; cursor: pointer;
  color: #64748b; border-bottom: 2px solid transparent;
  margin-bottom: -1px; user-select: none; white-space: nowrap; transition: color .15s;
}
.model-tab:hover { color: #94a3b8; }
.model-tab.active { color: #38bdf8; border-bottom-color: #38bdf8; }
.frame-container { flex: 1; position: relative; }
iframe { position: absolute; inset: 0; width: 100%; height: 100%; border: none; display: none; }
iframe.active { display: block; }
"""

INDEX_JS = """\
function switchModel(el) {
  document.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('iframe').forEach(f => f.classList.remove('active'));
  el.classList.add('active');
  var frame = document.getElementById(el.dataset.frame);
  frame.classList.add('active');
  if (!frame.src || frame.src === 'about:blank') frame.src = frame.dataset.src;
}
(function() { var first = document.querySelector('.model-tab'); if (first) switchModel(first); })();
"""


def build_html(model_group: str, group_data: dict[str, dict]) -> str:
    """Build one scaling report HTML page.

    group_data: {label: {(isl,osl,conc): {metric_key: float|None}}}
    """
    labels = sorted(group_data.keys())

    # Gather all (isl, osl) pairs as tab groups
    all_keys: set[tuple] = set()
    for pts in group_data.values():
        all_keys.update(pts.keys())

    isl_osl_pairs = sorted({(isl, osl) for isl, osl, _ in all_keys})

    # Palette assignment
    color_map = {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(labels)}
    # Build JS data blob: { "isl1024-osl1024": { label: { conc: { metric: val } } } }
    js_data: dict = {}
    for isl, osl in isl_osl_pairs:
        group_key = f"isl{isl}-osl{osl}"
        js_data[group_key] = {}
        for lbl in labels:
            pts = group_data[lbl]
            conc_map = {}
            for (i, o, c), metrics in pts.items():
                if i == isl and o == osl:
                    conc_map[c] = metrics
            if conc_map:
                js_data[group_key][lbl] = conc_map

    # Build tab bar
    tabs_html = ""
    panels_html = ""
    for idx, (isl, osl) in enumerate(isl_osl_pairs):
        tid = f"isl{isl}-osl{osl}"
        active = " active" if idx == 0 else ""
        tabs_html += f'<div class="tab{active}" data-panel="{tid}" onclick="switchTab(this)">ISL {isl} / OSL {osl}</div>\n'
        panels_html += f'<div id="panel-{tid}" class="tab-panel{active}" data-group="{tid}"></div>\n'

    colors_js = json.dumps(color_map)
    data_js = json.dumps(js_data)
    metrics_js = json.dumps([
        {"key": key, "label": label, "unit": unit, "hi": hi}
        for key, label, unit, hi in METRICS
    ])

    js = f"""
const LABELS   = {json.dumps(labels)};
const COLORS   = {colors_js};
const DATA     = {data_js};
const METRICS  = {metrics_js};

// ── Chart registry ────────────────────────────────────────────────────────────
const charts = {{}};

function destroyCharts(prefix) {{
  Object.keys(charts).filter(k => k.startsWith(prefix)).forEach(k => {{
    charts[k].destroy(); delete charts[k];
  }});
}}

// ── KPI strip ─────────────────────────────────────────────────────────────────
function buildKPI(groupKey, container) {{
  const gd = DATA[groupKey];
  if (!gd) return;

  // Use the first label as baseline; pick highest common concurrency
  const baseline = LABELS[0];
  const baseConcs = Object.keys(gd[baseline] || {{}}).map(Number);
  const conc = baseConcs.includes(64) ? 64 : Math.max(...baseConcs);

  let html = '';
  LABELS.forEach(lbl => {{
    const val = gd[lbl]?.[conc]?.output_throughput;
    if (val == null) return;
    const baseVal = gd[baseline]?.[conc]?.output_throughput;
    let cls = 'neu', delta = '';
    if (lbl !== baseline && baseVal) {{
      const pct = (val - baseVal) / baseVal * 100;
      cls = pct >= 0 ? 'pos' : 'neg';
      delta = `<div class="kpi-sub">${{pct >= 0 ? '+' : ''}}${{pct.toFixed(1)}}% vs ${{baseline}} @ conc ${{conc}}</div>`;
    }} else {{
      delta = `<div class="kpi-sub">@ concurrency ${{conc}}</div>`;
    }}
    html += `<div class="kpi-card">
      <div class="kpi-label">${{lbl}} — Peak Output Throughput</div>
      <div class="kpi-value ${{cls}}">${{val.toFixed(1)}}</div>
      <div class="kpi-sub">tok/s</div>${{delta}}
    </div>`;
  }});
  container.insertAdjacentHTML('beforeend', `<div class="kpi-strip">${{html}}</div>`);
}}

// ── Line chart builder ────────────────────────────────────────────────────────
function makeChart(canvasId, groupKey, metricKey, title, yLabel) {{
  const gd = DATA[groupKey];
  const concSet = new Set();
  LABELS.forEach(lbl => Object.keys(gd[lbl] || {{}}).forEach(c => concSet.add(Number(c))));
  const concs = Array.from(concSet).sort((a,b) => a - b);

  const datasets = LABELS.map(lbl => ({{
    label: lbl,
    data: concs.map(c => gd[lbl]?.[c]?.[metricKey] ?? null),
    borderColor: COLORS[lbl],
    backgroundColor: COLORS[lbl].replace(', 1)', ', 0.15)'),
    borderWidth: 2.5,
    pointRadius: 5,
    pointHoverRadius: 7,
    tension: 0.35,
    spanGaps: false,
  }}));

  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (charts[canvasId]) charts[canvasId].destroy();
  charts[canvasId] = new Chart(ctx, {{
    type: 'line',
    data: {{ labels: concs, datasets }},
    options: {{
      responsive: true, maintainAspectRatio: true,
      plugins: {{
        title: {{ display: false }},
        legend: {{ labels: {{ color: '#cbd5e1', font: {{ size: 12 }} }} }},
        tooltip: {{
          callbacks: {{
            label: ctx => `${{ctx.dataset.label}}: ${{ctx.parsed.y != null ? ctx.parsed.y.toFixed(2) : 'N/A'}} ${{yLabel}}`
          }}
        }}
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: 'Concurrency', color: '#94a3b8' }},
          ticks: {{ color: '#94a3b8' }},
          grid: {{ color: 'rgba(255,255,255,.07)' }},
        }},
        y: {{
          title: {{ display: true, text: yLabel, color: '#94a3b8' }},
          ticks: {{ color: '#94a3b8' }},
          grid: {{ color: 'rgba(255,255,255,.07)' }},
          beginAtZero: false,
        }}
      }}
    }}
  }});
}}

// ── Chart panels ─────────────────────────────────────────────────────────────
const CHART_GROUPS = [
  {{
    title: 'Throughput',
    charts: [
      {{ key: 'output_throughput',      label: 'Output Token Throughput', unit: 'tok/s' }},
      {{ key: 'request_throughput',     label: 'Request Throughput',      unit: 'req/s' }},
      {{ key: 'total_token_throughput', label: 'Total Token Throughput',  unit: 'tok/s' }},
    ]
  }},
  {{
    title: 'Time to First Token',
    charts: [
      {{ key: 'mean_ttft_ms',   label: 'Mean TTFT',   unit: 'ms' }},
      {{ key: 'median_ttft_ms', label: 'Median TTFT', unit: 'ms' }},
      {{ key: 'p99_ttft_ms',    label: 'P99 TTFT',    unit: 'ms' }},
    ]
  }},
  {{
    title: 'Time per Output Token',
    charts: [
      {{ key: 'mean_tpot_ms',   label: 'Mean TPOT',   unit: 'ms' }},
      {{ key: 'median_tpot_ms', label: 'Median TPOT', unit: 'ms' }},
      {{ key: 'p99_tpot_ms',    label: 'P99 TPOT',    unit: 'ms' }},
    ]
  }},
  {{
    title: 'Inter-token Latency',
    charts: [
      {{ key: 'mean_itl_ms',   label: 'Mean ITL',   unit: 'ms' }},
      {{ key: 'median_itl_ms', label: 'Median ITL', unit: 'ms' }},
      {{ key: 'p99_itl_ms',    label: 'P99 ITL',    unit: 'ms' }},
    ]
  }},
];

function buildCharts(groupKey, container) {{
  CHART_GROUPS.forEach(grp => {{
    let gridHtml = '';
    grp.charts.forEach(c => {{
      const cid = `${{groupKey}}-${{c.key}}`;
      gridHtml += `<div class="chart-card"><h3>${{c.label}}</h3><canvas id="${{cid}}"></canvas></div>`;
    }});
    container.insertAdjacentHTML('beforeend',
      `<h2 style="color:#667eea;font-size:.9rem;text-transform:uppercase;letter-spacing:.08em;margin:.5rem 0 .75rem">${{grp.title}}</h2>
       <div class="chart-grid">${{gridHtml}}</div>`);
    grp.charts.forEach(c => {{
      makeChart(`${{groupKey}}-${{c.key}}`, groupKey, c.key, c.label, c.unit);
    }});
  }});
}}

// ── Detailed table ────────────────────────────────────────────────────────────
function buildTable(groupKey, container) {{
  const gd = DATA[groupKey];
  const concSet = new Set();
  LABELS.forEach(lbl => Object.keys(gd[lbl] || {{}}).forEach(c => concSet.add(Number(c))));
  const concs = Array.from(concSet).sort((a,b) => a - b);

  let rows = '';
  LABELS.forEach(lbl => {{
    concs.forEach(c => {{
      const m = gd[lbl]?.[c];
      if (!m) return;
      const f = v => v != null ? v.toFixed(2) : '—';
      rows += `<tr>
        <td style="color:${{COLORS[lbl]}};font-weight:600">${{lbl}}</td>
        <td>${{c}}</td>
        <td>${{f(m.output_throughput)}}</td>
        <td>${{f(m.request_throughput)}}</td>
        <td>${{f(m.mean_ttft_ms)}}</td>
        <td>${{f(m.mean_tpot_ms)}}</td>
        <td>${{f(m.p99_ttft_ms)}}</td>
        <td>${{f(m.p99_tpot_ms)}}</td>
        <td>${{f(m.mean_itl_ms)}}</td>
      </tr>`;
    }});
  }});

  container.insertAdjacentHTML('beforeend', `
    <div class="table-card">
      <h3>Detailed Metrics</h3>
      <table>
        <thead><tr>
          <th>Variant</th><th>Concurrency</th>
          <th>Out tok/s</th><th>Req/s</th>
          <th>Mean TTFT</th><th>Mean TPOT</th>
          <th>P99 TTFT</th><th>P99 TPOT</th>
          <th>Mean ITL</th>
        </tr></thead>
        <tbody>${{rows}}</tbody>
      </table>
    </div>`);
}}

// ── Panel builder (lazy) ──────────────────────────────────────────────────────
function buildPanel(panel) {{
  if (panel.dataset.built) return;
  const groupKey = panel.dataset.group;
  buildKPI(groupKey, panel);
  buildCharts(groupKey, panel);
  buildTable(groupKey, panel);
  panel.dataset.built = '1';
}}

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(el) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  const panel = document.getElementById('panel-' + el.dataset.panel);
  panel.classList.add('active');
  buildPanel(panel);
}}

// Build first tab on load
(function() {{
  const firstPanel = document.querySelector('.tab-panel.active');
  if (firstPanel) buildPanel(firstPanel);
}})();
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Scaling Report — {model_group}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
{CSS}
  </style>
</head>
<body>

<header>
  <h1>Throughput vs Concurrency</h1>
  <p class="subtitle">Model: {model_group} &nbsp;·&nbsp; Variants: {', '.join(labels)}</p>
</header>

<div class="tab-bar">
{tabs_html}</div>

{panels_html}

<script>
{js}
</script>
</body>
</html>
"""
    return html


def build_index(reports: list[tuple[str, str]]) -> str:
    tabs = ""
    frames = ""
    for i, (label, fname) in enumerate(reports):
        fid = f"frame-{i}"
        tabs   += f'  <div class="model-tab" data-frame="{fid}" onclick="switchModel(this)">{label}</div>\n'
        frames += f'  <iframe id="{fid}" data-src="{fname}" src="about:blank"></iframe>\n'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Scaling Reports</title>
  <style>
{INDEX_CSS}
  </style>
</head>
<body>
<div class="header">
  <h1>Throughput vs Concurrency — Scaling Reports</h1>
  <div class="model-tab-bar">
{tabs}  </div>
</div>
<div class="frame-container">
{frames}</div>
<script>
{INDEX_JS}
</script>
</body>
</html>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR
    if not results_dir.exists():
        sys.exit(f"results/ not found at {results_dir}")

    print(f"Scanning {results_dir} …")
    groups = collect_groups(results_dir)
    if not groups:
        sys.exit("No bench JSON files found under results/")

    reports: list[tuple[str, str]] = []

    for model_group, group_data in sorted(groups.items()):
        if not group_data:
            continue
        html = build_html(model_group, group_data)
        fname = f"scaling-{model_group}.html"
        (OUT_DIR / fname).write_text(html)
        total_pts = sum(len(pts) for pts in group_data.values())
        print(f"  wrote {fname}  ({len(group_data)} variant(s), {total_pts} total points)")
        reports.append((model_group, fname))

    # Pick up any pre-existing scaling-*.html not produced this run
    existing = {fname for _, fname in reports}
    for p in sorted(OUT_DIR.glob("scaling-*.html")):
        if p.name not in existing and p.name != "scaling-index.html":
            reports.append((p.stem.removeprefix("scaling-"), p.name))

    if reports:
        (OUT_DIR / "scaling-index.html").write_text(build_index(reports))
        print(f"  wrote scaling-index.html  ({len(reports)} model(s))")


if __name__ == "__main__":
    main()
