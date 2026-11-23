"""Part 4: behaviour cloning on KUKA demos only, deployed on all four arms."""
from _common import eval_tasks, np, plt, savefig, update_summary, write_csv
from xembody.bc import collect, train
from xembody.env import NAMES, Arm, run_expert, run_policy

N_DEMOS, DART = 60, 0.005
data, ok, tries = collect("kuka_iiwa", N_DEMOS, seed=1, noise=DART)
print(f"{ok} KUKA demos ({len(data)} state-action pairs) from {tries} attempts")
pol, hist = {}, {}
for kind in ("agnostic", "proprio"):
    pol[kind], hist[kind] = train(data, kind)


def rate(res):
    return round(float(np.mean([r["success"] for r in res])), 2)


tasks = eval_tasks()
rows = []
for n in NAMES:
    arm = Arm(n)
    row = {"robot": n,
           "expert": rate([run_expert(arm, c, g) for c, g in tasks]),
           "bc_agnostic": rate([run_policy(arm, pol["agnostic"], c, g) for c, g in tasks]),
           "bc_proprio": rate([run_policy(arm, pol["proprio"], c, g) for c, g in tasks])}
    rows.append(row)
    print(row)
    arm.close()
write_csv("04_learn_once.csv", rows)
update_summary("part4", {"demos": ok, "pairs": len(data), "dart_sigma_m": DART, "rows": rows,
                         "final_train_mse": {k: v[-1] for k, v in hist.items()}})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2, 1]})
x, w = np.arange(len(NAMES)), 0.27
for k, (key, lab, col) in enumerate((("expert", "scripted expert", "0.6"),
                                     ("bc_agnostic", "BC-agnostic", "tab:green"),
                                     ("bc_proprio", "BC-proprio", "tab:purple"))):
    bars = a1.bar(x + (k - 1) * w, [100 * r[key] for r in rows], w, color=col, label=lab)
    a1.bar_label(bars, fmt="%.0f", fontsize=7)
a1.set_xticks(x, [n + (" (source)" if n == "kuka_iiwa" else "") for n in NAMES])
a1.set_ylabel(f"success rate (%), {len(tasks)} tasks"); a1.set_ylim(0, 110)
a1.legend(fontsize=8); a1.set_title("Learn on KUKA, deploy on every arm"); a1.grid(axis="y", alpha=0.3)
for kind, col in (("agnostic", "tab:green"), ("proprio", "tab:purple")):
    a2.semilogy(hist[kind], color=col, label=f"BC-{kind}")
a2.set_xlabel("epoch"); a2.set_ylabel("training MSE (log scale)")
a2.legend(fontsize=8); a2.grid(alpha=0.3); a2.set_title("Training loss on KUKA data")
savefig(fig, "04_learn_once_deploy_many.png")
