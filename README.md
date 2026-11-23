# AISR Laboratory (CSAI4014): Experiments 1–8

* **Student:** Aniruddh Vijayvargia
* **SAP ID:** 500125613
* **Roll no.:** R2142231769
* **Batch:** D (B.Tech CSE AI & ML, B4)
* **Partner:** none (performed individually)
* **Course teacher:** Dr. Resham Raj Shivwanshi

This repository holds the code, raw data and figures behind my laboratory report. Every
table and figure in the report comes from a run in this repository.

| Folder | Experiment | Sources |
| --- | --- | --- |
| `exp01-signals/` | 1. Signals, sampling, DFT and a simulated pendulum | Lab Manual Ch. 2 |
| `exp02-queens/` | 2. Eight queens as a constraint satisfaction problem | Lab Manual Ch. 3 |
| `exp03-search/` | 3. BFS and DFS path finding on a grid maze | Lab Manual Ch. 4 |
| `exp04-hillclimb/` | 4. Hill climbing and random restarts | Lab Manual Ch. 5 |
| `exp05-rl/` | 5. Tabular Q-learning in a grid world | Lab Manual Ch. 6 |
| `exp06-neural/` | 6. XOR with a 2-4-1 network and backpropagation | Lab Manual Ch. 7 |
| `exp07-webots/` | 7. Webots e-puck swarm and Pioneer 3-DX lidar navigation | github.com/reshamrajs/webots-robotics-sim |
| `exp08-xembody/` | 8. Cross-embodiment transfer of a pick-and-place skill (PyBullet) | self-written, see `exp08-xembody/NOTES.md` |

## Reproducing

* **Experiments 1–6.** Each folder has `expNN.py`, the source with `# %%` cells. It also has
  `expNN.ipynb`, the executed notebook made with `tools/py2nb.py`. Each run writes `figs/`
  and `results.json`.
  * Software: Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.2.

  ```
  python tools/py2nb.py exp01-signals/exp01.py
  ```

* **Experiment 7.** Needs Webots R2025a and the course repository. See `exp07-webots/NOTES.md`.
* **Experiment 8.** Needs the conda-forge env `aisr-xembody`, with PyBullet. See
  `exp08-xembody/NOTES.md`.

Hardware: Windows 11 Home, AMD Ryzen 7 5800HS.
