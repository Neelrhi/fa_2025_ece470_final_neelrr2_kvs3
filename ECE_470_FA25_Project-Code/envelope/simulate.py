import numpy as np
import itertools, csv, os, time
from .geometry import on_unit_circle, any_collision_with_obstacles, any_outside_unit_disk, any_robot_collision
from .config import CONFIG, OBSTACLES

def targets_from_permutation(n, sigma, radius=1.0):
    thetas = [((s+1) * np.pi / n + np.pi) for s in sigma]
    T = np.stack([on_unit_circle(t) * radius for t in thetas], axis=0)
    return T

def starts_on_unit_circle(n):
    thetas = [((k+1) * np.pi / n) for k in range(n)]  # k=1..n in spec
    X0 = np.stack([on_unit_circle(t) for t in thetas], axis=0)
    return X0

def speed_cap(V, vmax, eps=1e-12):
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    scale = np.minimum(1.0, vmax / (norms + eps))
    return V * scale


def _progress_iter(iterable, total, show, every):
    """
    If tqdm is available and show=True, wrap the iterable with a progress bar.
    Otherwise, yield and occasionally print simple progress.
    """
    if not show:
        for x in iterable:
            yield x
        return

    try:
        from tqdm import tqdm  # optional
        yield from tqdm(iterable, total=total)
    except Exception:
        # simple fallback printer
        count = 0
        for x in iterable:
            count += 1
            if every and (count % every == 0 or count == total):
                print(f"[progress] {count}/{total} permutations")
            yield x

def simulate_once(compute_gradients_fn, n, sigma, r, dt, vmax, tol, max_steps):
    X = starts_on_unit_circle(n)  # shape (n,2)
    # T = targets_from_permutation(n, sigma)  # shape (n,2)
    T = targets_from_permutation(n, sigma, radius=1.0 - 1e-3)

    t0 = time.time()
    log = [X.copy()]

    # Quick feasibility assert
    if any_collision_with_obstacles(X, OBSTACLES, r) or any_outside_unit_disk(X):
        return {"success": False, "reason": "bad_start", "steps": 0, "trace": log}
    
    for step in range(1, max_steps+1):
        V = compute_gradients_fn(X, T, OBSTACLES, r)  # expected shape (n,2)
        if not np.all(np.isfinite(V)):
            return {"success": False, "reason": "nan_in_student_output", "steps": step, "trace": [log[-1]]}
        V = speed_cap(np.asarray(V, dtype=float), vmax)
        X = X + dt * V

        # Check constraints
        if any_outside_unit_disk(X):
            return {"success": False, "reason": "left_unit_disk", "steps": step, "trace": log}
        if any_collision_with_obstacles(X, OBSTACLES, r):
            return {"success": False, "reason": "hit_obstacle", "steps": step, "trace": log}
        if any_robot_collision(X, r):
            return {"success": False, "reason": "robot_collision", "steps": step, "trace": log}

        log.append(X.copy())

        # Check done
        if np.all(np.linalg.norm(X - T, axis=1) <= tol):
            t1 = time.time()
            return {"success": True, "reason": "done", "steps": step, "elapsed": t1 - t0, "trace": log}

    return {"success": False, "reason": "timeout", "steps": max_steps, "trace": log}

def evaluate_all_permutations(compute_gradients_fn, n=5, r=0.05, dt=0.005, vmax=0.05, tol=0.005, max_steps=30000, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    traces_dir = os.path.join(results_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    all_perms = list(itertools.permutations(range(n)))
    records = []
    disqualified = False
    successes = 0
    total_steps = 0

    total = len(all_perms)
    show = bool(CONFIG.get("show_progress", True))
    every = int(CONFIG.get("progress_every", 5))

    for idx, sigma in enumerate(_progress_iter(all_perms, total, show, every), 0):
        out = simulate_once(compute_gradients_fn, n, sigma, r, dt, vmax, tol, max_steps)
        rec = {"perm_idx": idx, "sigma": list(sigma), "success": out.get("success", False),
               "reason": out.get("reason",""), "steps": out.get("steps", None)}
        
        records.append(rec)

        # Save the full trace for visualization
        trace_arr = np.asarray(out.get("trace", []), dtype=float)

        payload = {
            "perm_idx": idx,
            "sigma": list(sigma),
            "n": n,
            "r": r,
            "dt": dt,
            "vmax": vmax,
            "tol": tol,
            "max_steps": max_steps,
            "success": out.get("success", False),
            "reason": out.get("reason", ""),
            "steps": out.get("steps", None),
            "trace": trace_arr.tolist(),   # shape [T][n][2]
        }

        trace_path = os.path.join(traces_dir, f"perm_{idx:03d}.json")
        import json
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        if rec["success"]:
            successes += 1
            total_steps += rec["steps"]
        else:
            disqualified = True  # any failure disqualifies

    avg_steps = (total_steps / successes) if successes > 0 else None

    csv_path = os.path.join(results_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["perm_idx","sigma","success","reason","steps"])
        w.writeheader()
        for r in records:
            w.writerow(r)

    return {
        "n": n,
        "r": r,
        "dt": dt,
        "vmax": vmax,
        "tol": tol,
        "max_steps": max_steps,
        "successes": successes,
        "total_perms": len(all_perms),
        "avg_steps_over_successes": avg_steps,
        "disqualified": disqualified,
        "csv": csv_path,
    }

