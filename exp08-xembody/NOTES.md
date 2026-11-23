# Experiment 8: Cross-embodiment transfer of a pick-and-place skill (PyBullet)

**The scripts here are my own.** I did not have the faculty archive
`AISR-Cross-Embodiment-PyBullet`. I rebuilt the experiment in PyBullet from the
description in the report template (Parts 1–5, the four arms, and the eight analysis
questions), so the numbers will differ from the faculty version.

Environment: micromamba / conda-forge env `aisr-xembody`, Python 3.11.18, PyBullet API 202010061
(pybullet 3.25 from conda-forge), NumPy 2.4.6, Matplotlib 3.11.2, Windows 11, AMD Ryzen 7 5800HS.

```
micromamba create -n aisr-xembody -c conda-forge python=3.11 pybullet numpy matplotlib
cd exp08-xembody
python experiments/00_check_setup.py          # prints SETUP OK
python experiments/01_explore_embodiments.py  # results/01_embodiments.csv, 01_workspace.png
python experiments/02_joint_space_transfer.py # results/02_joint_space.csv, 02_joint_space_paths.png
python experiments/03_task_space_transfer.py  # results/03_task_space.csv, 03_task_space_paths.png
python experiments/04_learn_once_deploy_many.py
python experiments/05_multi_embodiment.py
python exercises/ex1_dart_noise.py
python exercises/ex2_scale_sweep.py
```

## Design (`xembody/env.py`, `xembody/bc.py`)

* **Arms.**

  | Arm | URDF (pybullet_data) | Notes |
  | --- | --- | --- |
  | kuka_iiwa | `kuka_iiwa/model.urdf` | 7 DOF |
  | franka_panda | `franka_panda/panda.urdf` | 7 DOF, grasp-target link |
  | xarm6 | `xarm/xarm6_robot.urdf` | 6 DOF |
  | kuka_mini | `kuka_iiwa/model.urdf`, `globalScaling = 0.6` | 7 DOF, same joints, shorter links |

* **Task.** Pick a 5 cm cube from a random spot at x 0.35–0.55 m, y 0.05–0.20 m and
  place it at a random goal at y −0.20 to −0.05 m.
  * Success means the cube rests on the floor within 3 cm of the goal.
  * Episodes are capped at 150 steps.
  * The evaluation uses the same 30 tasks on every arm (seed 100).
* **Common action space.** The end effector moves at most 2 cm per axis per step, plus a
  gripper command. Each arm reaches the target with PyBullet's position-only IK.
  * The 7-DOF arms also get a rest pose (null-space target) and joint limits.
  * Joints are set kinematically (`resetJointState`).
* **Gripper abstraction.** "Close" attaches the cube if the end effector is within 3 cm of
  it. "Open" releases it, and it falls under gravity.
  * Robot–cube collisions are off, so finger geometry plays no part.
* **Expert.** A scripted state machine in task space: above the cube, down, close, lift,
  above the goal, down, open. It retries if the grasp failed.
* **Behaviour cloning.** A NumPy MLP (2 × 64 tanh, Adam, MSE, 300 epochs) trained on
  successful expert demos.
  * The demos are collected with DART noise: σ = 5 mm is added to the executed motion, while
    the label stays the clean expert action.
  * **BC-agnostic** observation: cube − ee, goal − ee, ee height, holding flag.
  * **BC-proprio** adds the joint angles, zero-padded to 7.
* **Part 2 (joint replay).** The KUKA expert's joint vector at each step is copied to the
  other arm: the first 6 values for the xArm6, clipped to the joint limits.
* **Part 5.** Each test leaves one arm out.
  * **single:** 60 demos from one other arm. That is kuka_iiwa, or franka_panda when the
    iiwa is the unseen arm.
  * **multi:** 20 demos from each of the other three arms.
