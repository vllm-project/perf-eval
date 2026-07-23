#!/usr/bin/env python3
"""Generate one HTML benchmark report per model directory under ./results,
plus a benchmark-index.html wrapper for tabbing between models."""

import os
import re
import sys
import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"
OUT_DIR = Path(__file__).parent

# ── Metric extraction ─────────────────────────────────────────────────────────

METRIC_PATTERNS = {
    "Request throughput":      r"Request throughput \(req/s\):\s+([\d.]+)",
    "Output token throughput": r"Output token throughput \(tok/s\):\s+([\d.]+)",
    "Total token throughput":  r"Total token throughput \(tok/s\):\s+([\d.]+)",
    "Benchmark duration":      r"Benchmark duration \(s\):\s+([\d.]+)",
    "Mean TTFT":               r"Mean TTFT \(ms\):\s+([\d.]+)",
    "Median TTFT":             r"Median TTFT \(ms\):\s+([\d.]+)",
    "P99 TTFT":                r"P99 TTFT \(ms\):\s+([\d.]+)",
    "Mean TPOT":               r"Mean TPOT \(ms\):\s+([\d.]+)",
    "Median TPOT":             r"Median TPOT \(ms\):\s+([\d.]+)",
    "P99 TPOT":                r"P99 TPOT \(ms\):\s+([\d.]+)",
    "Mean ITL":                r"Mean ITL \(ms\):\s+([\d.]+)",
    "Median ITL":              r"Median ITL \(ms\):\s+([\d.]+)",
    "P99 ITL":                 r"P99 ITL \(ms\):\s+([\d.]+)",
}

HEADER_PATTERN = re.compile(
    r"attention_backend:\s*(\S+).*?isl:\s*(\d+)\s+osl:\s*(\d+)\s+conc:\s*(\d+)",
    re.DOTALL,
)
MOE_HEADER_PATTERN = re.compile(
    r"moe_backend:\s*(\S+).*?isl:\s*(\d+)\s+osl:\s*(\d+)\s+conc:\s*(\d+)",
    re.DOTALL,
)
OVERRIDE_PATTERN = re.compile(r"Overriding with (\S+) out of potential backends")


def parse_txt(path: Path) -> dict | None:
    text = path.read_text(errors="replace")
    m = HEADER_PATTERN.search(text)
    if not m:
        return None
    backend, isl, osl, conc = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    override = OVERRIDE_PATTERN.search(text)
    if override:
        backend = override.group(1)
    metrics = {}
    for name, pat in METRIC_PATTERNS.items():
        hit = re.search(pat, text)
        metrics[name] = float(hit.group(1)) if hit else None
    return {"backend": backend, "isl": isl, "osl": osl, "conc": conc, "metrics": metrics}


def parse_txt_moe(path: Path) -> dict | None:
    text = path.read_text(errors="replace")
    m = MOE_HEADER_PATTERN.search(text)
    if not m:
        return None
    backend, isl, osl, conc = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    metrics = {}
    for name, pat in METRIC_PATTERNS.items():
        hit = re.search(pat, text)
        metrics[name] = float(hit.group(1)) if hit else None
    return {"backend": backend, "isl": isl, "osl": osl, "conc": conc, "metrics": metrics}


# ── Directory scan ────────────────────────────────────────────────────────────

def collect_model_dir(model_dir: Path) -> tuple[dict, str | None]:
    """Return ({backend -> {(isl,osl,conc) -> metrics_dict}}, default_backend_name)

    The default backend is the one whose results live at the top level of the
    model directory (not inside an attn-* subdirectory).
    """
    data = defaultdict(dict)
    default_backend: str | None = None

    def ingest(txt_path: Path, backend_override: str | None = None) -> str | None:
        rec = parse_txt(txt_path)
        if rec is None:
            return None
        backend = backend_override or rec["backend"]
        key = (rec["isl"], rec["osl"], rec["conc"])
        data[backend][key] = rec["metrics"]
        return backend

    # top-level summary files — these are the default backend runs
    for txt in sorted(model_dir.glob("*-summary.txt")):
        name = ingest(txt)
        if name and default_backend is None:
            default_backend = name

    # attn-BACKEND subdirs — explicit backend sweep results
    for sub in sorted(model_dir.iterdir()):
        if sub.is_dir() and sub.name.startswith("attn-"):
            backend_name = sub.name[len("attn-"):]
            if backend_name == "default":
                attn_file = sub / "attn_backend.txt"
                if attn_file.exists():
                    raw = attn_file.read_text()
                    m = OVERRIDE_PATTERN.search(raw)
                    if m:
                        backend_name = m.group(1)
                    else:
                        # Fall back to the last non-empty word on the last line
                        # (covers "Using XYZ backend" style lines).
                        last = raw.strip().splitlines()[-1].strip()
                        backend_name = last.split()[0] if last else "default"
                # backend_name may still be "default" if attn_backend.txt is absent.
            for txt in sorted(sub.glob("*-summary.txt")):
                ingest(txt, backend_override=backend_name)

    return dict(data), default_backend


def collect_model_dir_moe(model_dir: Path) -> tuple[dict, str | None]:
    """Return ({backend -> {(isl,osl,conc) -> metrics_dict}}, default_backend_name)

    Looks for moe-* subdirectories instead of attn-* subdirectories.
    """
    data = defaultdict(dict)
    default_backend: str | None = None

    def ingest(txt_path: Path, backend_override: str | None = None) -> str | None:
        rec = parse_txt_moe(txt_path)
        if rec is None:
            return None
        backend = backend_override or rec["backend"]
        key = (rec["isl"], rec["osl"], rec["conc"])
        data[backend][key] = rec["metrics"]
        return backend

    # moe-BACKEND subdirs — explicit moe backend sweep results
    for sub in sorted(model_dir.iterdir()):
        if sub.is_dir() and sub.name.startswith("moe-"):
            backend_name = sub.name[len("moe-"):]
            if backend_name == "default":
                moe_file = sub / "moe_backend.txt"
                if moe_file.exists():
                    raw = moe_file.read_text()
                    # Look for vLLM's moe-specific log line, e.g.:
                    #   Using AITER Fp8 MoE backend out of potential backends: [...]
                    m = re.search(r"Using (.+?) MoE backend", raw)
                    if m:
                        backend_name = m.group(1)

                # backend_name stays "default" if the moe backend line is absent.
            for txt in sorted(sub.glob("*-summary.txt")):
                name = ingest(txt, backend_override=backend_name)
                if name and default_backend is None:
                    default_backend = name

    return dict(data), default_backend


# ── HTML generation ───────────────────────────────────────────────────────────

CSS = """
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: ui-monospace, "Cascadia Code", "SF Mono", Menlo, Consolas, monospace;
      background: #0f1117;
      color: #e2e8f0;
      padding: 2rem;
      font-size: 14px;
      line-height: 1.5;
    }

    h1 {
      font-size: 1.35rem;
      font-weight: 600;
      color: #f8fafc;
      margin-bottom: 0.3rem;
      letter-spacing: -0.01em;
    }

    .subtitle {
      color: #64748b;
      font-size: 0.82rem;
      margin-bottom: 1.75rem;
    }

    .legend {
      display: flex;
      gap: 1.5rem;
      margin-bottom: 1.5rem;
      font-size: 0.75rem;
      color: #64748b;
      align-items: center;
    }
    .legend-item { display: flex; align-items: center; gap: 0.4rem; }
    .swatch { width: 10px; height: 10px; border-radius: 2px; }
    .swatch.green { background: #22c55e; }
    .swatch.red   { background: #ef4444; }

    .tab-bar {
      display: flex;
      gap: 0;
      margin-bottom: 1.5rem;
      border-bottom: 1px solid #1e293b;
    }
    .tab {
      padding: 0.5rem 1.25rem;
      font-size: 0.8rem;
      cursor: pointer;
      color: #64748b;
      border-bottom: 2px solid transparent;
      margin-bottom: -1px;
      user-select: none;
      transition: color 0.15s;
    }
    .tab:hover { color: #94a3b8; }
    .tab.active { color: #38bdf8; border-bottom-color: #38bdf8; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }

    .conc-group { margin-bottom: 2.25rem; }
    .conc-label {
      font-size: 0.7rem;
      font-weight: 700;
      color: #475569;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 0.6rem;
    }

    table { width: 100%; border-collapse: collapse; }
    colgroup col:first-child { width: 13rem; }

    th, td { padding: 0.48rem 0.9rem; text-align: right; }
    th:first-child, td:first-child { text-align: left; }

    thead tr { border-bottom: 1px solid #1e293b; }
    thead th {
      font-size: 0.72rem;
      font-weight: 500;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding-bottom: 0.6rem;
    }
    thead th.backend-name {
      font-size: 0.82rem;
      text-transform: none;
      letter-spacing: normal;
      color: #cbd5e1;
      font-weight: 600;
    }
    thead th.backend-name.default-backend { color: #38bdf8; }

    tr.section-row td {
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #334155;
      padding-top: 0.9rem;
      padding-bottom: 0.25rem;
      border-bottom: none;
    }

    tbody tr { border-bottom: 1px solid #0f172a; }
    tbody tr:last-child { border-bottom: none; }
    tbody tr:not(.section-row):hover { background: #131c2e; }

    td.metric-label { color: #94a3b8; font-size: 0.8rem; padding-left: 1.5rem; }
    td.val { color: #cbd5e1; }
    td.val .unit { color: #334155; font-size: 0.68rem; margin-left: 0.2rem; }
    td.val.best  { color: #22c55e; font-weight: 700; }
    td.val.worst { color: #ef4444; }

    .delta { font-size: 0.68rem; margin-left: 0.3rem; opacity: 0.85; }
    .delta.good { color: #22c55e; }
    .delta.bad  { color: #ef4444; }
    .delta.neut { color: #475569; }
"""

JS_TEMPLATE = """
var BACKENDS = {backends_json};

var METRIC_DEFS = [
  {{ label: 'Request throughput',      hi: true,  unit: 'req/s' }},
  {{ label: 'Output token throughput', hi: true,  unit: 'tok/s' }},
  {{ label: 'Total token throughput',  hi: true,  unit: 'tok/s' }},
  {{ label: 'Benchmark duration',      hi: false, unit: 's'     }},
];
var TTFT_DEFS = [
  {{ label: 'Mean TTFT',   hi: false, unit: 'ms' }},
  {{ label: 'Median TTFT', hi: false, unit: 'ms' }},
  {{ label: 'P99 TTFT',    hi: false, unit: 'ms' }},
];
var TPOT_DEFS = [
  {{ label: 'Mean TPOT',   hi: false, unit: 'ms' }},
  {{ label: 'Median TPOT', hi: false, unit: 'ms' }},
  {{ label: 'P99 TPOT',    hi: false, unit: 'ms' }},
];
var ITL_DEFS = [
  {{ label: 'Mean ITL',   hi: false, unit: 'ms' }},
  {{ label: 'Median ITL', hi: false, unit: 'ms' }},
  {{ label: 'P99 ITL',    hi: false, unit: 'ms' }},
];

var DATA = {data_json};

function fmt(v, unit) {{
  if (v === null || v === undefined) return '—';
  if (unit === 'ms' || unit === 's' || unit === 'req/s' || unit === 'tok/s') return v.toFixed(2);
  return String(v);
}}

function deltaHtml(base, val, hi) {{
  if (base === null || val === null) return '';
  var pct = (val - base) / base * 100;
  if (Math.abs(pct) < 0.1) return '<span class="delta neut">(~0%)</span>';
  var sign = pct > 0 ? '+' : '';
  var cls = hi ? (pct > 0 ? 'good' : 'bad') : (pct < 0 ? 'good' : 'bad');
  return '<span class="delta ' + cls + '">(' + sign + pct.toFixed(1) + '%)</span>';
}}

function buildTable(key, concLabel) {{
  var rows = DATA[key];
  if (!rows) return '<p style="color:#475569;font-size:0.8rem">No data for ' + key + '</p>';

  var sections = [
    {{ label: 'Throughput',            defs: METRIC_DEFS }},
    {{ label: 'Time to First Token',   defs: TTFT_DEFS   }},
    {{ label: 'Time per Output Token', defs: TPOT_DEFS   }},
    {{ label: 'Inter-token Latency',   defs: ITL_DEFS    }},
  ];

  var headCols = '<th>Metric</th>';
  BACKENDS.forEach(function(b) {{
    var cls = 'backend-name' + (b.isDefault ? ' default-backend' : '');
    var tag = b.isDefault ? ' <span style="color:#334155;font-size:0.65rem;font-weight:400">(default)</span>' : '';
    headCols += '<th class="' + cls + '">' + b.label + tag + '</th>';
  }});

  var bodyHtml = '';
  sections.forEach(function(sec) {{
    bodyHtml += '<tr class="section-row"><td colspan="' + (BACKENDS.length + 1) + '">' + sec.label + '</td></tr>';
    sec.defs.forEach(function(def) {{
      var vals = rows[def.label];
      if (!vals) return;
      var valid = vals.map(function(v, i) {{ return {{ v: v, i: i }}; }}).filter(function(x) {{ return x.v !== null; }});
      var bestIdx = null, worstIdx = null;
      if (valid.length > 1) {{
        var sorted = valid.slice().sort(function(a, b) {{ return def.hi ? b.v - a.v : a.v - b.v; }});
        bestIdx  = sorted[0].i;
        worstIdx = sorted[sorted.length - 1].i;
      }}
      var baseVal = vals[0];
      var cells = '<td class="metric-label">' + def.label + '</td>';
      vals.forEach(function(v, i) {{
        var cls = 'val';
        if (i === bestIdx)  cls += ' best';
        if (i === worstIdx) cls += ' worst';
        var delta = i === 0 ? '' : deltaHtml(baseVal, v, def.hi);
        cells += '<td class="' + cls + '">' + fmt(v, def.unit) + '<span class="unit">' + def.unit + '</span>' + delta + '</td>';
      }});
      bodyHtml += '<tr>' + cells + '</tr>';
    }});
  }});

  return '<table><colgroup><col/>' + BACKENDS.map(function() {{ return '<col/>'; }}).join('') + '</colgroup>'
       + '<thead><tr>' + headCols + '</tr></thead>'
       + '<tbody>' + bodyHtml + '</tbody></table>';
}}

function buildContent(islKey) {{
  var concKeys = Object.keys(DATA).filter(function(k) {{ return k.startsWith(islKey + '-'); }});
  concKeys.sort(function(a, b) {{
    var ca = parseInt(a.replace(/.*conc/, ''));
    var cb = parseInt(b.replace(/.*conc/, ''));
    return ca - cb;
  }});
  var html = '';
  concKeys.forEach(function(k) {{
    var conc = k.replace(/.*conc/, '');
    html += '<div class="conc-group"><div class="conc-label">Concurrency ' + conc + '</div>' + buildTable(k) + '</div>';
  }});
  return html || '<p style="color:#475569;font-size:0.8rem">No data</p>';
}}

function switchTab(el) {{
  document.querySelectorAll('.tab').forEach(function(t) {{ t.classList.remove('active'); }});
  document.querySelectorAll('.tab-content').forEach(function(c) {{ c.classList.remove('active'); }});
  el.classList.add('active');
  var target = document.getElementById(el.dataset.tab);
  target.classList.add('active');
  if (!target.dataset.built) {{
    target.innerHTML = buildContent(el.dataset.tab);
    target.dataset.built = '1';
  }}
}}

// Build first tab on load
(function() {{
  var firstTab = document.querySelector('.tab.active');
  if (firstTab) {{
    var target = document.getElementById(firstTab.dataset.tab);
    if (target) {{
      target.innerHTML = buildContent(firstTab.dataset.tab);
      target.dataset.built = '1';
    }}
  }}
}})();
"""


def build_html(model_name: str, backend_data: dict, default_backend: str | None) -> str:
    # Put the default backend first; sort the rest alphabetically
    all_backends = sorted(backend_data.keys())
    if default_backend and default_backend in all_backends:
        all_backends.remove(default_backend)
        all_backends = [default_backend] + all_backends

    # Collect all (isl,osl,conc) test points
    all_keys: set[tuple] = set()
    for bdata in backend_data.values():
        all_keys.update(bdata.keys())

    # Build DATA dict: { "isl1024-conc4": { "Mean TTFT": [v_backend0, v_backend1, ...], ... } }
    data_dict = {}
    for (isl, osl, conc) in sorted(all_keys):
        dk = f"isl{isl}-conc{conc}"
        row: dict[str, list] = {}
        for mname in METRIC_PATTERNS:
            vals = []
            for b in all_backends:
                bdata = backend_data.get(b, {})
                point = bdata.get((isl, osl, conc), {})
                vals.append(point.get(mname) if point else None)
            row[mname] = vals
        data_dict[dk] = row

    # Determine ISL groups for tabs
    isl_values = sorted({k[0] for k in all_keys})
    osl_values = sorted({k[1] for k in all_keys})
    osl_label = "/".join(str(o) for o in osl_values)

    backends_js = json.dumps([
        {"key": b.lower().replace("_", ""), "label": b, "isDefault": i == 0}
        for i, b in enumerate(all_backends)
    ], indent=2)

    data_js = json.dumps(data_dict, indent=2)

    # Tab bar HTML
    tabs_html = ""
    tab_divs = ""
    for i, isl in enumerate(isl_values):
        tab_id = f"isl{isl}"
        active = " active" if i == 0 else ""
        tabs_html += f'  <div class="tab{active}" data-tab="{tab_id}" onclick="switchTab(this)">ISL {isl} / OSL {osl_label}</div>\n'
        tab_divs += f'<div id="{tab_id}" class="tab-content{active}"></div>\n'

    subtitle = f"Model: {model_name}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Benchmark — {model_name}</title>
  <style>
{CSS}
  </style>
</head>
<body>

<h1>Attention Backend Benchmark</h1>
<p class="subtitle">{subtitle}</p>

<div class="legend">
  <div class="legend-item"><div class="swatch green"></div> Best in row</div>
  <div class="legend-item"><div class="swatch red"></div> Worst in row</div>
  <span style="color:#334155">· deltas relative to default (first) backend</span>
</div>

<div class="tab-bar">
{tabs_html}</div>

{tab_divs}
<script>
{JS_TEMPLATE.format(backends_json=backends_js, data_json=data_js)}
</script>
</body>
</html>
"""
    return html


# ── Index wrapper ─────────────────────────────────────────────────────────────

INDEX_CSS = """
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    html, body { height: 100%; }

    body {
      font-family: ui-monospace, "Cascadia Code", "SF Mono", Menlo, Consolas, monospace;
      background: #0f1117;
      color: #e2e8f0;
      font-size: 14px;
      display: flex;
      flex-direction: column;
    }

    .header {
      padding: 0.75rem 2rem 0;
      flex-shrink: 0;
    }

    h1 {
      font-size: 1.2rem;
      font-weight: 600;
      color: #f8fafc;
      margin-bottom: 0.6rem;
      letter-spacing: -0.01em;
    }

    .model-tab-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 0;
      border-bottom: 1px solid #1e293b;
    }

    .model-tab {
      padding: 0.5rem 1.4rem;
      font-size: 0.8rem;
      cursor: pointer;
      color: #64748b;
      border-bottom: 2px solid transparent;
      margin-bottom: -1px;
      user-select: none;
      transition: color 0.15s;
    }
    .model-tab:hover { color: #94a3b8; }
    .model-tab.active { color: #38bdf8; border-bottom-color: #38bdf8; }

    .frame-container {
      flex: 1;
      position: relative;
    }

    iframe {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      border: none;
      display: none;
    }
    iframe.active { display: block; }
"""

INDEX_JS = """
function switchModel(el) {
  document.querySelectorAll('.model-tab').forEach(function(t) { t.classList.remove('active'); });
  document.querySelectorAll('iframe').forEach(function(f) { f.classList.remove('active'); });
  el.classList.add('active');
  var frame = document.getElementById(el.dataset.frame);
  frame.classList.add('active');
  // lazy-load: only set src when first activated
  if (!frame.src || frame.src === 'about:blank') {
    frame.src = frame.dataset.src;
  }
}

// activate first tab on load
(function() {
  var first = document.querySelector('.model-tab');
  if (first) switchModel(first);
})();
"""


def build_index_html(reports: list[tuple[str, str]]) -> str:
    """Build a single-page wrapper that tabs between per-model report iframes.

    reports: list of (model_label, filename) sorted as desired.
    """
    tabs_html = ""
    frames_html = ""
    for i, (label, filename) in enumerate(reports):
        frame_id = f"frame-{i}"
        tabs_html += f'  <div class="model-tab" data-frame="{frame_id}" onclick="switchModel(this)">{label}</div>\n'
        frames_html += f'  <iframe id="{frame_id}" data-src="{filename}" src="about:blank"></iframe>\n'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Benchmark Reports</title>
  <style>
{INDEX_CSS}
  </style>
</head>
<body>

<div class="header">
  <h1>Attention Backend Benchmarks</h1>
  <div class="model-tab-bar">
{tabs_html}  </div>
</div>

<div class="frame-container">
{frames_html}</div>

<script>
{INDEX_JS}
</script>
</body>
</html>
"""


def build_html_moe(model_name: str, backend_data: dict, default_backend: str | None) -> str:
    """Build a per-model MoE backend benchmark HTML page."""
    all_backends = sorted(backend_data.keys())
    if default_backend and default_backend in all_backends:
        all_backends.remove(default_backend)
        all_backends = [default_backend] + all_backends

    all_keys: set[tuple] = set()
    for bdata in backend_data.values():
        all_keys.update(bdata.keys())

    data_dict = {}
    for (isl, osl, conc) in sorted(all_keys):
        dk = f"isl{isl}-conc{conc}"
        row: dict[str, list] = {}
        for mname in METRIC_PATTERNS:
            vals = []
            for b in all_backends:
                bdata = backend_data.get(b, {})
                point = bdata.get((isl, osl, conc), {})
                vals.append(point.get(mname) if point else None)
            row[mname] = vals
        data_dict[dk] = row

    isl_values = sorted({k[0] for k in all_keys})
    osl_values = sorted({k[1] for k in all_keys})
    osl_label = "/".join(str(o) for o in osl_values)

    backends_js = json.dumps([
        {"key": b.lower().replace("_", ""), "label": b, "isDefault": i == 0}
        for i, b in enumerate(all_backends)
    ], indent=2)

    data_js = json.dumps(data_dict, indent=2)

    tabs_html = ""
    tab_divs = ""
    for i, isl in enumerate(isl_values):
        tab_id = f"isl{isl}"
        active = " active" if i == 0 else ""
        tabs_html += f'  <div class="tab{active}" data-tab="{tab_id}" onclick="switchTab(this)">ISL {isl} / OSL {osl_label}</div>\n'
        tab_divs += f'<div id="{tab_id}" class="tab-content{active}"></div>\n'

    subtitle = f"Model: {model_name}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MoE Benchmark — {model_name}</title>
  <style>
{CSS}
  </style>
</head>
<body>

<h1>MoE Backend Benchmark</h1>
<p class="subtitle">{subtitle}</p>

<div class="legend">
  <div class="legend-item"><div class="swatch green"></div> Best in row</div>
  <div class="legend-item"><div class="swatch red"></div> Worst in row</div>
  <span style="color:#334155">· deltas relative to default (first) MoE backend</span>
</div>

<div class="tab-bar">
{tabs_html}</div>

{tab_divs}
<script>
{JS_TEMPLATE.format(backends_json=backends_js, data_json=data_js)}
</script>
</body>
</html>
"""
    return html


def build_index_html_moe(reports: list[tuple[str, str]]) -> str:
    """Build a single-page wrapper that tabs between per-model MoE report iframes."""
    tabs_html = ""
    frames_html = ""
    for i, (label, filename) in enumerate(reports):
        frame_id = f"frame-{i}"
        tabs_html += f'  <div class="model-tab" data-frame="{frame_id}" onclick="switchModel(this)">{label}</div>\n'
        frames_html += f'  <iframe id="{frame_id}" data-src="{filename}" src="about:blank"></iframe>\n'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MoE Backend Benchmark Reports</title>
  <style>
{INDEX_CSS}
  </style>
</head>
<body>

<div class="header">
  <h1>MoE Backend Benchmarks</h1>
  <div class="model-tab-bar">
{tabs_html}  </div>
</div>

<div class="frame-container">
{frames_html}</div>

<script>
{INDEX_JS}
</script>
</body>
</html>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def model_name_from_dir(dir_name: str) -> str:
    """Best-effort human label from directory name like attn_sweep_gpt_oss_120b_mi355x."""
    name = re.sub(r"^attn-sweep-", "", dir_name)
    return name


def model_name_from_dir_moe(dir_name: str) -> str:
    """Best-effort human label from directory name like moe_sweep_deepseek_r1_0528_mi355x."""
    name = re.sub(r"^moe-sweep-", "", dir_name)
    return name


def main():
    if not RESULTS_DIR.exists():
        sys.exit(f"results/ not found at {RESULTS_DIR}")

    model_dirs = [
        d for d in sorted(RESULTS_DIR.iterdir())
        if d.is_dir() and (d.name.startswith("attn-sweep") or d.name.startswith("moe-sweep"))
    ]
    if not model_dirs:
        sys.exit("No subdirectories found in results/")

    reports: list[tuple[str, str]] = []

    for model_dir in model_dirs:
        backend_data, default_backend = collect_model_dir(model_dir)
        if not backend_data:
            print(f"  skip {model_dir.name} — no parseable summary files")
            continue

        model_name = model_name_from_dir(model_dir.name)
        html = build_html(model_name, backend_data, default_backend)

        out_filename = f"benchmark-{model_dir.name}.html"
        out_path = OUT_DIR / out_filename
        out_path.write_text(html)
        backends = list(backend_data.keys())
        points = sum(len(v) for v in backend_data.values())
        print(f"  wrote {out_filename}  ({len(backends)} backends, {points} test points)")
        reports.append((model_name, out_filename))

    if reports:
        index_path = OUT_DIR / "benchmark-index.html"
        index_path.write_text(build_index_html(reports))
        print(f"  wrote benchmark-index.html  ({len(reports)} model(s))")

    # ── MoE backend sweep reports ─────────────────────────────────────────────
    moe_reports: list[tuple[str, str]] = []

    for model_dir in model_dirs:
        # Only process directories that contain at least one moe-* subdir
        has_moe = any(
            sub.is_dir() and sub.name.startswith("moe-")
            for sub in model_dir.iterdir()
        )
        if not has_moe:
            continue

        backend_data, default_backend = collect_model_dir_moe(model_dir)
        if not backend_data:
            print(f"  skip {model_dir.name} (moe) — no parseable summary files")
            continue

        model_name = model_name_from_dir_moe(model_dir.name)
        html = build_html_moe(model_name, backend_data, default_backend)

        out_filename = f"moe-benchmark-{model_dir.name}.html"
        out_path = OUT_DIR / out_filename
        out_path.write_text(html)
        backends = list(backend_data.keys())
        points = sum(len(v) for v in backend_data.values())
        print(f"  wrote {out_filename}  ({len(backends)} backends, {points} test points)")
        moe_reports.append((model_name, out_filename))

    if moe_reports:
        index_path = OUT_DIR / "moe-benchmark-index.html"
        index_path.write_text(build_index_html_moe(moe_reports))
        print(f"  wrote moe-benchmark-index.html  ({len(moe_reports)} model(s))")


if __name__ == "__main__":
    main()
