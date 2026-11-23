# Experiment 7: Webots swarm and lidar navigation (my runs)

Course repository: https://github.com/reshamrajs/webots-robotics-sim (cloned at commit 95c5db2).
Webots R2025a on Windows 11 Home, AMD Ryzen 7 5800HS, Python 3.11.9.

## What I changed

* No controller code was changed. All parameters were set from the command line with `--set`,
  so `controllers/` is not copied here.
* `tools/run_experiment.py` (diff in `tools/run_experiment.patch`):
  * **Windows fix.** The output path `C:\Users\...` was passed to `re.sub` as the replacement
    string, which failed with `re.error: bad escape \L`. I now use forward slashes and a
    function replacement.
  * **`--replace OLD|NEW`.** Edits the world text before the run. I used it to rotate the
    starting headings and to add lidar noise (Exercise 5).
  * **Per-run temporary world names.** Each run writes `_experiment_<label>.wbt`, so several
    runs can go in parallel.
* `tools/run_batch2.py` runs the perturbed-start and noise jobs three at a time.
  `tools/jobs.txt` and `tools/run_one.sh` ran the first batch.

## Commands

```
python tools/run_experiment.py swarm   --seconds 120 --name swarm_default_r1        # r2, r3 identical
python tools/run_experiment.py swarm   --seconds 120 --set cruise=0.9 --name swarm_cruise0.9_r1
python tools/run_experiment.py pioneer --seconds 180 --name pioneer_default_r1
python tools/run_experiment.py pioneer --seconds 180 --set bubble=0.15 --name pioneer_bubble0.15_r1
python tools/run_experiment.py pioneer --seconds 180 --set k_turn=0.5  --name pioneer_kturn0.5_r1
python tools/run_experiment.py swarm   --seconds 120 --set cruise=0.2 --name ex1_cruise0.2_r1   # 0.2 ... 1.0
python tools/run_batch2.py                                   # p2/p3 starts, ex5 lidar noise
python analysis/exp07_analysis.py                            # metrics, results.json, figs/
```

All runs used `--mode=fast --no-rendering --batch --minimize`. The raw CSVs (t, robot, x, y, yaw)
are in `runs/` and the console output of each run is in `logs/`.

## Finding: headless runs are deterministic

Three exact repeats of the same setting gave **bit-for-bit identical** CSVs, for both the swarm
and the Pioneer. The guide expects run-to-run variation, but there is none here (SD = 0).

To get a spread, repeats p2 and p3 rotate every e-puck's start heading by +0.02 rad and
−0.02 rad (about 1.1°). The mean ± SD values in `results.json` use r1, p2 and p3.

The Pioneer perturbed runs did not start. argparse read the negative angle in `--replace` as an
option, so the Pioneer rows use the identical exact repeats.

## Metrics

These follow the course `analysis/metrics.py`:

* Path length is the sum of the step distances.
* Mean speed is the path length divided by the time.
* Stuck % is the share of samples slower than 1 cm/s.
* Coverage is the share of grid cells visited: a 10 cm grid on the 1.6 m arena, and a 50 cm
  grid on the 8 m room.

## Figures

* `figs/fig7_1_webots_swarm_gui.png`: screenshot of the swarm world running in the Webots GUI
  on my laptop.
* `figs/fig7_2_swarm_trajectories.png`: swarm paths and coverage.
* `figs/fig7_3_pioneer_trajectory.png`: Pioneer path and coverage.
* `figs/fig7_4_exercise1_cruise.png`: Exercise 1 (cruise sweep).
* `figs/fig7_extra_pioneer_bubble015.png`: Pioneer with bubble = 0.15 m.
