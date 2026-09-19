import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
rows = []
for session in ("A1", "B1", "B2", "A2"):
    runs = []
    for rep in range(1, 6):
        p = root / session / f"run-{rep}.json"
        if not p.exists():
            raise RuntimeError(f"Missing result: {p}")
        d = json.loads(p.read_text())
        if d.get("completed") != 512 or d.get("failed", 0):
            raise RuntimeError(f"Incomplete benchmark: {p}")
        row = {
            "session": session, "arm": session[0], "repetition": rep,
            "ttft_ms": d["mean_ttft_ms"], "tpot_ms": d["mean_tpot_ms"],
            "total_tok_s_gpu": d["total_token_throughput"] / 8,
            "output_tok_s_gpu": d["output_throughput"] / 8,
        }
        rows.append(row)
        runs.append(row)
    print(session, json.dumps({
        k: statistics.median(r[k] for r in runs)
        for k in ("ttft_ms", "tpot_ms", "total_tok_s_gpu")
    }))

summary = {}
for scope, selected in (("all", rows), ("exclude_first_per_session", [r for r in rows if r["repetition"] > 1])):
    medians = {
        arm: {key: statistics.median(r[key] for r in selected if r["arm"] == arm)
              for key in ("ttft_ms", "tpot_ms", "total_tok_s_gpu", "output_tok_s_gpu")}
        for arm in ("A", "B")
    }
    summary[scope] = {"medians": medians, "B_vs_A_percent": {
        key: 100 * (medians["B"][key] / medians["A"][key] - 1)
        for key in medians["A"]
    }}
(root / "summary.json").write_text(json.dumps({"runs": rows, "summary": summary}, indent=2) + "\n")
print(json.dumps(summary, indent=2))
