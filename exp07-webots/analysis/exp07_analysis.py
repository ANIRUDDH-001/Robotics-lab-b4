"""Metrics, tables and figures for Experiment 7 from my own Webots run CSVs.

    python analysis/exp07_analysis.py        (run from exp07-webots/)

Reads runs/*.csv (columns t, robot, x, y, yaw written by the trajectory_logger
supervisor), uses the same metric definitions as the course repository's
analysis/metrics.py, and writes results.json and figs/*.png.
"""
import csv
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = {"swarm": (0.8, 0.1), "pioneer": (4.0, 0.5)}     # arena half-width (m), cell size (m)
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)


def load(path):
    tracks = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            tracks[r["robot"]].append((float(r["t"]), float(r["x"]), float(r["y"])))
    return tracks


def metrics(path, world):
    half, cell = GRID[world]
    n = int(round(2 * half / cell))
    tracks, visited, per = load(path), set(), {}
    for name, p in sorted(tracks.items()):
        length, stuck = 0.0, 0
        for (t0, x0, y0), (t1, x1, y1) in zip(p, p[1:]):
            d = math.hypot(x1 - x0, y1 - y0)
            length += d
            stuck += d / max(t1 - t0, 1e-9) < 0.01          # slower than 1 cm/s
        for _, x, y in p:
            visited.add((min(n - 1, int((x + half) / cell)), min(n - 1, int((y + half) / cell))))
        dur = p[-1][0] - p[0][0]
        per[name] = {"path": length, "speed": length / dur, "stuck": 100 * stuck / max(len(p) - 1, 1)}
    paths = [v["path"] for v in per.values()]
    return {"robots": per, "mean_path": float(np.mean(paths)),
            "mean_speed": float(np.mean([v["speed"] for v in per.values()])),
            "mean_stuck": float(np.mean([v["stuck"] for v in per.values()])),
            "worst_stuck": float(max(v["stuck"] for v in per.values())),
            "coverage": 100 * len(visited) / n ** 2, "duration": dur}


def runs(prefixes, world):
    out = []
    for pre in prefixes:
        for f in sorted(glob.glob(os.path.join(HERE, "runs", pre + ".csv"))):
            m = metrics(f, world)
            m["file"] = os.path.basename(f)
            out.append(m)
    return out


def summary(rs, key):
    v = np.array([r[key] for r in rs])
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


R = {}
# ---- Part A: swarm. r1 = nominal start; p2/p3 = start headings rotated by +/-0.02 rad.
for tag, pre in (("default", "swarm_default"), ("cruise0.9", "swarm_cruise0.9")):
    exact = runs([pre + "_r*"], "swarm")
    pert = [exact[0]] + runs([pre + "_p*"], "swarm")
    R[f"swarm_{tag}"] = {
        "exact_repeats_identical": len({round(r["coverage"], 6) for r in exact}) == 1
        and len({round(r["mean_path"], 6) for r in exact}) == 1,
        "runs": [{k: r[k] for k in ("file", "mean_path", "mean_speed", "mean_stuck", "worst_stuck", "coverage")}
                 for r in pert],
        "mean_sd": {k: summary(pert, k) for k in ("mean_path", "mean_speed", "mean_stuck", "worst_stuck", "coverage")},
    }

# ---- Part B: Pioneer (exact repeats; perturbation not used)
for tag in ("default", "bubble0.15", "kturn0.5"):
    rs = runs([f"pioneer_{tag}_r*"], "pioneer")
    R[f"pioneer_{tag}"] = {"n_runs": len(rs), "identical": len({round(r["mean_path"], 6) for r in rs}) == 1,
                           **{k: rs[0][k] for k in ("mean_path", "mean_speed", "worst_stuck", "coverage")}}

# ---- Exercise 1 (GUIDE 10.1): cruise sweep, 3 runs each (nominal + two perturbed starts)
sweep = []
for c in ("0.2", "0.4", "0.5", "0.6", "0.8", "1.0"):
    if c == "0.5":
        rs = [runs(["swarm_default_r1"], "swarm")[0]] + runs(["swarm_default_p*"], "swarm")
    else:
        rs = [runs([f"ex1_cruise{c}_r1"], "swarm")[0]] + runs([f"swarm_cruise{c}_p*"], "swarm")
    cov, wst = summary(rs, "coverage"), summary(rs, "worst_stuck")
    sweep.append({"cruise": float(c), "n": len(rs), "coverage": cov, "worst_stuck": wst,
                  "mean_path": summary(rs, "mean_path")})
R["exercise1_cruise_sweep"] = sweep

# ---- Exercise 5 (GUIDE 10.5): lidar noise on the Pioneer
R["exercise5_lidar_noise"] = [{"noise": 0.0, **{k: R["pioneer_default"][k] for k in ("mean_path", "worst_stuck", "coverage")}}]
for nz in ("0.001", "0.003"):
    m = runs([f"ex5_noise{nz}_r1"], "pioneer")[0]
    R["exercise5_lidar_noise"].append({"noise": float(nz), **{k: m[k] for k in ("mean_path", "worst_stuck", "coverage")}})

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(R, f, indent=2)
print(json.dumps({k: (v if k.startswith("ex") else v.get("mean_sd", v)) for k, v in R.items()}, indent=1)[:4000])


# ---------------- figures ----------------
def traj_plot(csv_name, world, out, title):
    half, cell = GRID[world]
    tracks = load(os.path.join(HERE, "runs", csv_name))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    allp = []
    for name, p in sorted(tracks.items()):
        p = np.array(p)[:, 1:]
        allp.append(p)
        a1.plot(p[:, 0], p[:, 1], lw=0.9, label=name)
        a1.plot(*p[0], "o", ms=5, color=a1.lines[-1].get_color())
        a1.plot(*p[-1], "x", ms=6, color=a1.lines[-1].get_color())
    a1.set(xlim=(-half, half), ylim=(-half, half), aspect="equal", xlabel="x (m)", ylabel="y (m)",
           title="Paths (o = start, x = end)")
    a1.legend(fontsize=7, loc="upper right"); a1.grid(alpha=0.3)
    allp = np.vstack(allp)
    bins = int(round(2 * half / cell))
    Hh, _, _ = np.histogram2d(allp[:, 0], allp[:, 1], bins=bins, range=[[-half, half], [-half, half]])
    a2.imshow((Hh.T > 0), origin="lower", extent=(-half, half, -half, half), cmap="Greens", vmin=0, vmax=1.3)
    a2.set(xlabel="x (m)", ylabel="y (m)", aspect="equal",
           title=f"Cells visited: {100 * (Hh > 0).mean():.0f}% of {cell * 100:.0f} cm grid")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", out), dpi=150); plt.close(fig)


traj_plot("swarm_default_r1.csv", "swarm", "fig7_2_swarm_trajectories.png",
          "e-puck swarm, default parameters, 120 s (my run swarm_default_r1)")
traj_plot("pioneer_default_r1.csv", "pioneer", "fig7_3_pioneer_trajectory.png",
          "Pioneer 3-DX follow-the-gap, default parameters, 180 s, 8 m x 8 m room")
traj_plot("pioneer_bubble0.15_r1.csv", "pioneer", "fig7_extra_pioneer_bubble015.png",
          "Pioneer 3-DX with bubble = 0.15 m, 180 s")

fig, ax1 = plt.subplots(figsize=(7.5, 3.8))
cr = [s["cruise"] for s in sweep]
ax1.errorbar(cr, [s["coverage"][0] for s in sweep], yerr=[s["coverage"][1] for s in sweep],
             fmt="o-", color="tab:green", capsize=3, label="coverage (mean ± SD, 3 runs)")
ax1.set_xlabel("cruise parameter (fraction of max wheel speed)"); ax1.set_ylabel("arena coverage (%)", color="tab:green")
ax1.set_ylim(0, 100); ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.errorbar(cr, [s["worst_stuck"][0] for s in sweep], yerr=[s["worst_stuck"][1] for s in sweep],
             fmt="s--", color="tab:red", capsize=3, label="worst-robot stuck (mean ± SD)")
ax2.set_ylabel("worst-robot time stuck (%)", color="tab:red"); ax2.set_ylim(0, 100)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
ax1.set_title("Exercise 1: speed against safety, 5 e-pucks, 120 s runs", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "fig7_4_exercise1_cruise.png"), dpi=150)
print("figures written")
