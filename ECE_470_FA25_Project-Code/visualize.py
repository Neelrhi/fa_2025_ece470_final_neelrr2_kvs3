# viz_results.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from envelope.config import CONFIG, OBSTACLES
from envelope.simulate import starts_on_unit_circle, targets_from_permutation

# Select which permutations to plot (0-based index for convenience)
PERM_SELECTION = list(range(120)) 
# PERM_SELECTION = [8, 14, 50] 
# PERM_SELECTION = [116, 117, 118] 
# PERM_SELECTION =  [0, 1]

def _draw_environment(ax, obstacles, r):
    unit_circle = plt.Circle((0, 0), 1.0, fill=False, linewidth=1.5, linestyle="-", alpha=0.7)
    ax.add_patch(unit_circle)
    for obs in obstacles:
        c = obs["center"]; R = obs["radius"]
        ax.add_patch(plt.Circle(c, R, color="black", alpha=0.12, linewidth=1.0))
        ax.plot([c[0]], [c[1]], marker="x", markersize=4)
    # show r as a little ruler
    ax.plot([0, r], [0, 0], linestyle="--", linewidth=1.0)

def plot_trace(payload, out_png):
    n = payload["n"]; r = payload["r"]
    robot_R = r 
    sigma = payload["sigma"]
    trace = np.asarray(payload["trace"], dtype=float)  # (T, n, 2)

    starts = starts_on_unit_circle(n)
    targets = targets_from_permutation(n, sigma)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    _draw_environment(ax, OBSTACLES, r)

    if trace.size > 0:
        for k in range(n):
            path = trace[:, k, :]
            ax.plot(path[:, 0], path[:, 1], linewidth=1.2)
            ax.plot([path[0, 0]], [path[0, 1]], marker="o", markersize=3)     # start marker
            ax.plot([path[-1, 0]], [path[-1, 1]], marker="s", markersize=3)   # end marker
            end_circle = patches.Circle((path[-1, 0], path[-1, 1]),
                                    radius=robot_R, fill=False,
                                    linewidth=1.0, linestyle="--", alpha=0.8)
            ax.add_patch(end_circle)

    ax.scatter(starts[:, 0], starts[:, 1], s=12, marker="o", label="starts")
    ax.scatter(targets[:, 0], targets[:, 1], s=12, marker="x", label="targets")

    steps = max(0, trace.shape[0] - 1)
    title = f"perm={tuple(sigma)} | steps={steps} | success={payload['success']} ({payload['reason']})"
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.25); ax.legend(loc="upper right", fontsize=8)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def main():
    results_dir = "results"
    traces_dir = os.path.join(results_dir, "traces")
    figs_dir = os.path.join(results_dir, "figures")
    os.makedirs(figs_dir, exist_ok=True)

    for idx in PERM_SELECTION:
        # Runner saves files numbered 1..120
        jpath = os.path.join(traces_dir, f"perm_{idx:03d}.json")
        if not os.path.exists(jpath):
            print(f"[skip] missing: {jpath}")
            continue

        with open(jpath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        out_png = os.path.join(figs_dir, f"perm_{idx:03d}.png")
        plot_trace(payload, out_png)
        print(f"[ok] saved: {out_png}")

if __name__ == "__main__":
    main()
