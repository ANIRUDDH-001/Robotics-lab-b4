"""Shared paths and helpers for the Experiment 8 scripts (run them from exp08-xembody/)."""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"savefig.dpi": 150, "font.size": 9})
N_EVAL, EVAL_SEED = 30, 100
COLORS = {"kuka_iiwa": "tab:blue", "franka_panda": "tab:orange", "xarm6": "tab:green", "kuka_mini": "tab:red"}


def eval_tasks(n=N_EVAL, seed=EVAL_SEED):
    from xembody.env import sample_task
    rng = np.random.default_rng(seed)
    return [sample_task(rng) for _ in range(n)]


def write_csv(name, rows):
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("wrote", path)


def update_summary(key, value):
    path = os.path.join(RESULTS, "summary.json")
    data = json.load(open(path)) if os.path.exists(path) else {}
    data[key] = value
    json.dump(data, open(path, "w"), indent=2)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, name))
    plt.close(fig)
    print("wrote", os.path.join(RESULTS, name))
