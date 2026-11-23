"""Exercise 1: how much DART noise does behaviour cloning need?

BC-agnostic is trained on 60 KUKA demos collected with noise sigma in {0, 2, 5, 10, 20} mm
and evaluated on the source arm (kuka_iiwa) and on an unseen arm (xarm6).
Run from exp08-xembody/:  python exercises/ex1_dart_noise.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from _common import eval_tasks, np, plt, savefig, update_summary, write_csv  # noqa: E402
from xembody.bc import collect, train  # noqa: E402
from xembody.env import Arm, run_policy  # noqa: E402

SIGMAS = [0.0, 0.002, 0.005, 0.01, 0.02]
EVAL_ON = ["kuka_iiwa", "xarm6"]
tasks = eval_tasks()
arms = {n: Arm(n) for n in EVAL_ON}
rows = []
for s in SIGMAS:
    data, ok, tries = collect("kuka_iiwa", 60, seed=1, noise=s, max_tries=300)
    pol, _ = train(data, "agnostic")
    row = {"dart_sigma_mm": round(1000 * s, 1), "demos": ok, "pairs": len(data)}
    for n in EVAL_ON:
        row[n] = round(float(np.mean([run_policy(arms[n], pol, c, g)["success"] for c, g in tasks])), 2)
    rows.append(row)
    print(row)
write_csv("ex1_dart_noise.csv", rows)
update_summary("exercise1", rows)

fig, ax = plt.subplots(figsize=(6.5, 3.5))
for n, m in zip(EVAL_ON, ["o-", "s--"]):
    ax.plot([r["dart_sigma_mm"] for r in rows], [100 * r[n] for r in rows], m, label=f"deployed on {n}")
ax.set_xlabel("DART noise sigma during data collection (mm per step)")
ax.set_ylabel(f"BC-agnostic success (%), {len(tasks)} tasks")
ax.set_ylim(0, 105); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax.set_title("Exercise 1: noise injection against covariate shift")
savefig(fig, "ex1_dart_noise.png")
