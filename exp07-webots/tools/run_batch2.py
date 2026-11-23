"""Second batch of runs: perturbed-start repeats and the lidar-noise exercise.

Exact repeats of a setting were bit-for-bit identical in --batch fast mode on
this machine, so repeats 2 and 3 of each swarm setting rotate every robot's
starting heading by +0.02 rad and -0.02 rad (about 1.1 degrees).
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

EPUCK_ROT = {1: "3.9", 2: "-0.8", 3: "0.8", 4: "2.3", 5: "-1.6"}
PIONEER_ROT = "-2.1708948203565805"
NOISE_OLD = "translation 0.136 0 0.059"


def epuck_rot(delta):
    reps = []
    for i, a in EPUCK_ROT.items():
        old = f'rotation 0 0 1 {a} name "e-puck{i}"'
        new = f'rotation 0 0 1 {float(a) + delta:.4f} name "e-puck{i}"'
        reps.append(f"{old}|{new}")
    return reps


jobs = []
for setting in [None, "cruise=0.9", "cruise=0.2", "cruise=0.4", "cruise=0.6", "cruise=0.8", "cruise=1.0"]:
    tag = "default" if setting is None else setting.replace("=", "")
    for k, d in (("p2", 0.02), ("p3", -0.02)):
        args = ["swarm", "--seconds", "120", "--name", f"swarm_{tag}_{k}", "--replace", *epuck_rot(d)]
        if setting:
            args += ["--set", setting]
        jobs.append(args)
for k, d in (("p2", 0.02), ("p3", -0.02)):
    jobs.append(["pioneer", "--seconds", "180", "--name", f"pioneer_default_{k}",
                 "--replace", f"{PIONEER_ROT}|{float(PIONEER_ROT) + d:.6f}"])
for nz in ("0.001", "0.003"):
    jobs.append(["pioneer", "--seconds", "180", "--name", f"ex5_noise{nz}_r1",
                 "--replace", f"{NOISE_OLD}|{NOISE_OLD} noise {nz}"])


def finished(args):
    name = args[args.index("--name") + 1]
    try:
        return "coverage" in open(f"logs/{name}.log").read()
    except OSError:
        return False


jobs = [j for j in jobs if not finished(j)]
print("jobs to run:", [j[j.index("--name") + 1] for j in jobs], flush=True)


def run(args):
    name = args[args.index("--name") + 1]
    with open(f"logs/{name}.log", "w") as fh:
        subprocess.run([sys.executable, "tools/run_experiment.py", *args], stdout=fh, stderr=subprocess.STDOUT)
    return name


with ThreadPoolExecutor(3) as ex:
    for n in ex.map(run, jobs):
        print("done", n, flush=True)
