"""Part 2: record KUKA expert joint trajectories and replay the same joint angles on the other arms."""
from _common import COLORS, eval_tasks, np, plt, savefig, update_summary, write_csv
from xembody.env import NAMES, SUCCESS_DIST, Arm, run_expert

N = 20
tasks = eval_tasks(N)
kuka = Arm("kuka_iiwa")
demos = []
for cube, goal in tasks:                             # record q_t, grip_t and the KUKA EE path
    r = run_expert(kuka, cube, goal, record=True)
    qs = [ob["q"] for ob, _ in r["log"]][1:] + [kuka.q()]
    grips = [a[3] for _, a in r["log"]]
    demos.append((cube, goal, qs, grips, r["path"][1:]))
kuka.close()

rows, paths = [], {}
for n in NAMES:
    arm = Arm(n)
    succ, devs, errs = 0, [], []
    for k, (cube, goal, qs, grips, ref) in enumerate(demos):
        arm.reset(cube, goal)
        path = []
        for q, g in zip(qs, grips):
            arm.set_q(np.clip(np.asarray(q)[:arm.dof], arm.lo, arm.hi))   # same numbers, different arm
            arm.apply_grip(g)
            path.append(arm.ee_pos())
        path = np.array(path)
        devs.append(float(np.mean(np.linalg.norm(path - ref, axis=1))))
        err, floor = arm.goal_error()
        errs.append(err)
        succ += bool(floor and err < SUCCESS_DIST and not arm.held)
        if k == 0:
            paths[n] = (path, ref, cube, goal)
    rows.append({"robot": n, "success_rate": round(succ / N, 2),
                 "mean_path_dev_cm": round(100 * float(np.mean(devs)), 1),
                 "median_err_cm": round(100 * float(np.median(errs)), 1)})
    print(rows[-1])
    arm.close()
write_csv("02_joint_space.csv", rows)
update_summary("part2", rows)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
cube, goal = paths["kuka_iiwa"][2], paths["kuka_iiwa"][3]
for n in NAMES:
    pth = paths[n][0]
    ls = "-" if n == "kuka_iiwa" else "--"
    a1.plot(pth[:, 0], pth[:, 1], ls, color=COLORS[n], label=n)
    a2.plot(pth[:, 0], pth[:, 2], ls, color=COLORS[n], label=n)
for a, (i, j) in ((a1, (0, 1)), (a2, (0, 2))):
    a.plot(cube[i], cube[j], "s", ms=9, color="k", label="cube start")
    a.plot(goal[i], goal[j], "*", ms=13, color="gold", mec="k", label="goal")
    a.grid(alpha=0.3)
    a.set_xlabel("x (m)")
a1.set_ylabel("y (m)"); a1.set_title("Top view: same joint angles on every arm"); a1.axis("equal")
a2.set_ylabel("z (m)"); a2.set_title("Side view"); a2.legend(fontsize=7)
savefig(fig, "02_joint_space_paths.png")
