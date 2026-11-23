"""Part 1: workspace and reach of each arm, and how much of the task region it can reach."""
from _common import COLORS, np, plt, savefig, update_summary, write_csv
from xembody.env import APPROACH, CUBE_X, CUBE_Y, CUBE_Z, GOAL_X, GOAL_Y, NAMES, Arm

N_SAMPLES = 3000
# task points: 5 x 4 grid of cube positions and of goal positions, at floor and at approach height
grid = []
for (xr, yr) in ((CUBE_X, CUBE_Y), (GOAL_X, GOAL_Y)):
    for x in np.linspace(*xr, 5):
        for y in np.linspace(*yr, 4):
            for z in (CUBE_Z, CUBE_Z + APPROACH):
                grid.append(np.array([x, y, z]))

rows, clouds = [], {}
rng = np.random.default_rng(0)
for n in NAMES:
    arm = Arm(n)
    pts = []
    for _ in range(N_SAMPLES):                       # forward kinematics of random joint vectors
        arm.set_q(rng.uniform(arm.lo, arm.hi))
        pts.append(arm.ee_pos())
    pts = np.array(pts)
    clouds[n] = pts
    reach = float(np.max(np.linalg.norm(pts, axis=1)))
    ok = 0
    for g in grid:                                   # inverse kinematics from the rest pose
        arm.set_q(arm.rest)
        arm.ik(g, iters=5)
        ok += np.linalg.norm(arm.ee_pos() - g) < 0.01
    rows.append({"robot": n, "dof": arm.dof, "max_reach_m": round(reach, 3),
                 "task_coverage_pct": round(100 * ok / len(grid), 1)})
    print(rows[-1])
    arm.close()
write_csv("01_embodiments.csv", rows)
update_summary("part1", rows)

fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
for ax, n in zip(axs, NAMES):
    pts = clouds[n]
    r = np.hypot(pts[:, 0], pts[:, 1])
    ax.scatter(r, pts[:, 2], s=1, alpha=0.25, color=COLORS[n], label="sampled EE positions")
    ax.add_patch(plt.Rectangle((0.35, 0.0), np.hypot(0.55, 0.2) - 0.35, CUBE_Z + APPROACH, fill=False,
                               ec="k", lw=1.2, label="task region"))
    ax.set_title(f"{n} ({rows[NAMES.index(n)]['task_coverage_pct']:.0f}% of task points)")
    ax.set_xlabel("horizontal distance from base (m)")
    ax.set_xlim(0, 1.4); ax.set_ylim(-0.6, 1.5); ax.grid(alpha=0.3)
axs[0].set_ylabel("height z (m)")
axs[0].legend(fontsize=7, loc="upper right", markerscale=6)
savefig(fig, "01_workspace.png")
