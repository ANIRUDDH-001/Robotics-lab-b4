"""Part 3: run the task-space expert (EE targets + each arm's own IK) on every arm."""
from _common import COLORS, eval_tasks, np, plt, savefig, update_summary, write_csv
from xembody.env import NAMES, Arm, run_expert

tasks = eval_tasks()
rows, paths = [], {}
for n in NAMES:
    arm = Arm(n)
    res = [run_expert(arm, c, g) for c, g in tasks]
    ok = [r for r in res if r["success"]]
    rows.append({"robot": n, "success_rate": round(len(ok) / len(res), 2),
                 "mean_steps_success": round(float(np.mean([r["steps"] for r in ok])), 1) if ok else "",
                 "median_err_cm": round(100 * float(np.median([r["err"] for r in res])), 1)})
    paths[n] = res[0]["path"]
    print(rows[-1])
    arm.close()
write_csv("03_task_space.csv", rows)
update_summary("part3", rows)

cube, goal = tasks[0]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
for n in NAMES:
    pth = paths[n]
    a1.plot(pth[:, 0], pth[:, 1], color=COLORS[n], label=n)
    a2.plot(pth[:, 0], pth[:, 2], color=COLORS[n], label=n)
for a, (i, j) in ((a1, (0, 1)), (a2, (0, 2))):
    a.plot(cube[i], cube[j], "s", ms=9, color="k", label="cube start")
    a.plot(goal[i], goal[j], "*", ms=13, color="gold", mec="k", label="goal")
    a.grid(alpha=0.3)
    a.set_xlabel("x (m)")
a1.set_ylabel("y (m)"); a1.set_title("Top view: same EE targets, each arm's own IK"); a1.axis("equal")
a2.set_ylabel("z (m)"); a2.set_title("Side view"); a2.legend(fontsize=7)
savefig(fig, "03_task_space_paths.png")
