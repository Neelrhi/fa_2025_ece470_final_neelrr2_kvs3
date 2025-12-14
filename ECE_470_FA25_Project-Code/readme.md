# Final Project — Swarm Navigation

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run (simulate all permutations)
```bash
python run_all.py
```
Outputs in `results/`:
- `summary.csv` – success/fail, reason, steps
- `runs/perm_###.json` – small per-permutation metadata
- `traces/perm_###.json` – pretty JSON trajectory (0-based indices)

## Visualize saved runs (no simulation)
```bash
python visualize.py
```
Writes PNGs to `results/figures/perm_###.png`.

Can edit which pngs are created by setting PERM_SELECTION in `visualize.py`.

## Rules enforced
- **Boundary (unit circle):** center-only ⇒ fail if ‖x‖ > 1 (‖x‖ == 1 allowed).
- **Obstacles:** outer-edge ⇒ collide if ‖x − c_obs‖ ≤ R_obs + r.
- **Robot–Robot:** outer-edge ⇒ collide if ‖x_i − x_j‖ ≤ 2r.

## Knobs (config.py)
```python
CONFIG = {
  "n": 5, "r": 0.05, "dt": 0.005, "vmax": 0.05, "tol": 0.005,
  "max_steps": 30000, "results_dir": "results",
  "show_progress": True, "progress_every": 5,
}
```

##  Clean re-run:
```bash
rm -rf results && python run_all.py && python visualize.py
```
