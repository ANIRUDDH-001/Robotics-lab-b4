"""Part 0: check that PyBullet loads all four arms and the expert runs."""
import _common  # noqa: F401  (sets the import path)
import numpy as np
import pybullet as p
from xembody.env import NAMES, Arm, run_expert, sample_task

print("PyBullet API", p.getAPIVersion(), "| NumPy", np.__version__)
for n in NAMES:
    arm = Arm(n)
    r = run_expert(arm, *sample_task(np.random.default_rng(0)))
    print(f"{n:13s} DOF={arm.dof}  expert success={r['success']}  steps={r['steps']}")
    arm.close()
print("SETUP OK")
