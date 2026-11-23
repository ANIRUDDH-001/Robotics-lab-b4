"""Part 5: leave one arm out; train on one other arm (single) or on all the others (multi), same data size."""
from _common import eval_tasks, np, plt, savefig, update_summary, write_csv
from xembody.bc import collect, train
from xembody.env import NAMES, Arm, run_policy

TOTAL = 60                      # demos per training set, single or multi
UNSEEN = ["kuka_iiwa", "franka_panda", "xarm6"]
tasks = eval_tasks()
cache = {}


def demos(name, n):
    if (name, n) not in cache:
        cache[(name, n)] = collect(name, n, seed=10 + NAMES.index(name), max_tries=6 * n)
        print(f"  demos {name}: {cache[(name, n)][1]} successes from {cache[(name, n)][2]} tries")
    return cache[(name, n)][0]


rows = []
for un in UNSEEN:
    others = [n for n in NAMES if n != un]
    single_src = "kuka_iiwa" if un != "kuka_iiwa" else "franka_panda"
    sets = {"single": demos(single_src, TOTAL),
            "multi": sum((demos(n, TOTAL // len(others)) for n in others), [])}
    arm = Arm(un)
    row = {"unseen": un, "single_source": single_src, "multi_sources": "+".join(others)}
    for kind in ("agnostic", "proprio"):
        for s in ("single", "multi"):
            pol, _ = train(sets[s], kind)
            row[f"{s}_{kind}"] = round(float(np.mean([run_policy(arm, pol, c, g)["success"] for c, g in tasks])), 2)
    arm.close()
    rows.append(row)
    print(row)
write_csv("05_multi_embodiment.csv", rows)
update_summary("part5", rows)

cols = [("single_agnostic", "single, agnostic", "#a1d99b"), ("multi_agnostic", "multi, agnostic", "tab:green"),
        ("single_proprio", "single, proprio", "#bcbddc"), ("multi_proprio", "multi, proprio", "tab:purple")]
fig, ax = plt.subplots(figsize=(8.5, 3.8))
x, w = np.arange(len(UNSEEN)), 0.2
for k, (key, lab, col) in enumerate(cols):
    bars = ax.bar(x + (k - 1.5) * w, [100 * r[key] for r in rows], w, color=col, label=lab)
    ax.bar_label(bars, fmt="%.0f", fontsize=7)
ax.set_xticks(x, [f"unseen: {u}" for u in UNSEEN]); ax.set_ylim(0, 115)
ax.set_ylabel(f"success on the unseen arm (%), {len(tasks)} tasks"); ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=8, ncol=4, loc="upper center")
ax.set_title(f"Leave-one-embodiment-out, {TOTAL} demos per training set")
savefig(fig, "05_multi_embodiment.png")
