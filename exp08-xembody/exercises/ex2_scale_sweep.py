"""Exercise 2: where does the capability gap start?

The KUKA iiwa is loaded with globalScaling 0.5 ... 1.0 and the task-space expert is run on the
same 30 tasks. Also checks the IK rest pose (null-space target) by switching it off on the full-size arm.
Run from exp08-xembody/:  python exercises/ex2_scale_sweep.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from _common import eval_tasks, np, plt, savefig, update_summary, write_csv  # noqa: E402
from xembody.env import Arm, run_expert  # noqa: E402

tasks = eval_tasks()
rows = []
for s in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    arm = Arm("kuka_iiwa", scale=s)
    res = [run_expert(arm, c, g) for c, g in tasks]
    rows.append({"scale": s, "success_rate": round(float(np.mean([r["success"] for r in res])), 2),
                 "median_err_cm": round(100 * float(np.median([r["err"] for r in res])), 1)})
    print(rows[-1])
    arm.close()
write_csv("ex2_scale_sweep.csv", rows)

rest = []
for name in ("kuka_iiwa", "franka_panda"):
    for use in (True, False):
        arm = Arm(name, rest_pose=use)
        res = [run_expert(arm, c, g) for c, g in tasks]
        ok = [r for r in res if r["success"]]
        rest.append({"robot": name, "rest_pose": use, "success_rate": round(len(ok) / len(res), 2),
                     "mean_steps_success": round(float(np.mean([r["steps"] for r in ok])), 1) if ok else None})
        print(rest[-1])
        arm.close()
update_summary("exercise2", {"scale": rows, "rest_pose": rest})

fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot([r["scale"] for r in rows], [100 * r["success_rate"] for r in rows], "o-", color="tab:red")
ax.set_xlabel("KUKA iiwa size (globalScaling, 1.0 = real arm)")
ax.set_ylabel(f"task-space expert success (%), {len(tasks)} tasks")
ax.set_ylim(0, 105); ax.grid(alpha=0.3)
ax.set_title("Exercise 2: capability gap as the arm shrinks")
savefig(fig, "ex2_scale_sweep.png")
