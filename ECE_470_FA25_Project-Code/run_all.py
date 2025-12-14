import argparse, importlib, os, json
import numpy as np

from envelope.simulate import evaluate_all_permutations

def load_student_compute():
    mod = importlib.import_module("student_code.my_nav_fn")
    if not hasattr(mod, "compute_gradients"):
        raise RuntimeError("student_code.my_nav_fn must define compute_gradients(...)")
    return mod.compute_gradients

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--r", type=float, default=0.05)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--vmax", type=float, default=0.05)
    p.add_argument("--tol", type=float, default=0.005)
    p.add_argument("--max_steps", type=int, default=30000)
    args = p.parse_args()

    compute = load_student_compute()
    results = evaluate_all_permutations(
        compute_gradients_fn=compute,
        n=args.n, r=args.r, dt=args.dt, vmax=args.vmax, tol=args.tol, max_steps=args.max_steps
    )
    # print(json.dumps(results, indent=2))
    print(f"Summary CSV: {results['csv']}")
    if results["disqualified"]:
        print("NOTE: at least one permutation failed.")

if __name__ == "__main__":
    main()
