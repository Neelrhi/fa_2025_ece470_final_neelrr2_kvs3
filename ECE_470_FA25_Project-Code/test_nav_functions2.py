"""
Round 2: Iterate on the best performers to find even faster solutions.
"""
import numpy as np
import time
from envelope.simulate import simulate_once

TEST_PERMS = [
    (0,1,2,3,4), (4,3,2,1,0), (2,0,4,1,3), 
    (1,3,0,4,2), (3,4,1,2,0), (0,2,4,1,3)
]

def test_algorithm(compute_fn, name, max_steps=30000):
    print(f"\nTesting: {name}")
    successes, total_steps = 0, 0
    
    for sigma in TEST_PERMS:
        result = simulate_once(compute_fn, n=5, sigma=sigma, r=0.05, 
                               dt=0.005, vmax=0.05, tol=0.005, max_steps=max_steps)
        if result.get('success'):
            successes += 1
            total_steps += result.get('steps')
    
    avg_steps = total_steps / successes if successes > 0 else float('inf')
    print(f"  Result: {successes}/6 success, avg={avg_steps:.0f} steps")
    return successes, avg_steps


# ============================================================================
# Based on Linear Attraction (best performer) - try variations
# ============================================================================

def v1_linear_base(state, targets, obstacles, r):
    """Base linear attraction (from round 1)."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0
    K_OBS, K_ROBOT, K_WALL = 0.01, 0.01, 0.0002
    D_OBS, D_ROBOT, D_WALL = 0.15, 0.15, 0.005
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        F = K_ATT * (target - pos)  # Linear attraction
        
        for obs in obstacles:
            c = np.array(obs["center"])
            v = pos - c
            dc = np.linalg.norm(v)
            ds = dc - obs["radius"] - r
            if D_OBS > ds > EPS:
                F += K_OBS * (1/ds - 1/D_OBS)**2 * v/dc
        
        for j in range(n):
            if i != j:
                v = pos - state[j]
                dc = np.linalg.norm(v)
                ds = dc - 2*r
                if D_ROBOT > ds > EPS:
                    F += K_ROBOT * (1/ds - 1/D_ROBOT)**2 * v/dc
        
        do = np.linalg.norm(pos)
        dw = 1.0 - do
        if D_WALL > dw > EPS and do > EPS:
            F += K_WALL * (1/dw - 1/D_WALL)**2 * (-pos/do)
        
        velocities[i] = F
    return velocities


def v2_stronger_linear(state, targets, obstacles, r):
    """Even stronger linear attraction."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 8.0  # Increased
    K_OBS, K_ROBOT, K_WALL = 0.015, 0.015, 0.0003
    D_OBS, D_ROBOT, D_WALL = 0.12, 0.12, 0.004
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        F = K_ATT * (target - pos)
        
        for obs in obstacles:
            c = np.array(obs["center"])
            v = pos - c
            dc = np.linalg.norm(v)
            ds = dc - obs["radius"] - r
            if D_OBS > ds > EPS:
                F += K_OBS * (1/ds - 1/D_OBS)**2 * v/dc
        
        for j in range(n):
            if i != j:
                v = pos - state[j]
                dc = np.linalg.norm(v)
                ds = dc - 2*r
                if D_ROBOT > ds > EPS:
                    F += K_ROBOT * (1/ds - 1/D_ROBOT)**2 * v/dc
        
        do = np.linalg.norm(pos)
        dw = 1.0 - do
        if D_WALL > dw > EPS and do > EPS:
            F += K_WALL * (1/dw - 1/D_WALL)**2 * (-pos/do)
        
        velocities[i] = F
    return velocities


def v3_very_strong_linear(state, targets, obstacles, r):
    """Very strong linear attraction, minimal repulsion."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 10.0
    K_OBS, K_ROBOT, K_WALL = 0.008, 0.008, 0.00015
    D_OBS, D_ROBOT, D_WALL = 0.1, 0.1, 0.003
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        F = K_ATT * (target - pos)
        
        for obs in obstacles:
            c = np.array(obs["center"])
            v = pos - c
            dc = np.linalg.norm(v)
            ds = dc - obs["radius"] - r
            if D_OBS > ds > EPS:
                F += K_OBS * (1/ds - 1/D_OBS)**2 * v/dc
        
        for j in range(n):
            if i != j:
                v = pos - state[j]
                dc = np.linalg.norm(v)
                ds = dc - 2*r
                if D_ROBOT > ds > EPS:
                    F += K_ROBOT * (1/ds - 1/D_ROBOT)**2 * v/dc
        
        do = np.linalg.norm(pos)
        dw = 1.0 - do
        if D_WALL > dw > EPS and do > EPS:
            F += K_WALL * (1/dw - 1/D_WALL)**2 * (-pos/do)
        
        velocities[i] = F
    return velocities


def v4_hybrid_linear_const(state, targets, obstacles, r):
    """Hybrid: Linear far, constant near (for final approach)."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 6.0
    K_OBS, K_ROBOT, K_WALL = 0.01, 0.01, 0.0002
    D_OBS, D_ROBOT, D_WALL = 0.15, 0.15, 0.005
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        vec = target - pos
        d = np.linalg.norm(vec)
        
        # Hybrid: use constant magnitude when close, linear when far
        if d > 0.1:
            F = K_ATT * vec  # Linear
        else:
            F = K_ATT * 0.1 * vec / (d + EPS)  # Constant magnitude
        
        for obs in obstacles:
            c = np.array(obs["center"])
            v = pos - c
            dc = np.linalg.norm(v)
            ds = dc - obs["radius"] - r
            if D_OBS > ds > EPS:
                F += K_OBS * (1/ds - 1/D_OBS)**2 * v/dc
        
        for j in range(n):
            if i != j:
                v = pos - state[j]
                dc = np.linalg.norm(v)
                ds = dc - 2*r
                if D_ROBOT > ds > EPS:
                    F += K_ROBOT * (1/ds - 1/D_ROBOT)**2 * v/dc
        
        do = np.linalg.norm(pos)
        dw = 1.0 - do
        if D_WALL > dw > EPS and do > EPS:
            F += K_WALL * (1/dw - 1/D_WALL)**2 * (-pos/do)
        
        velocities[i] = F
    return velocities


def v5_linear_cubic_repulsion(state, targets, obstacles, r):
    """Linear attraction with cubic (softer) repulsion."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 6.0
    K_OBS, K_ROBOT, K_WALL = 0.0005, 0.0005, 0.00001
    D_OBS, D_ROBOT, D_WALL = 0.2, 0.2, 0.01
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        F = K_ATT * (target - pos)
        
        for obs in obstacles:
            c = np.array(obs["center"])
            v = pos - c
            dc = np.linalg.norm(v)
            ds = dc - obs["radius"] - r
            if D_OBS > ds > EPS:
                F += K_OBS * (1/ds - 1/D_OBS)**3 * v/dc  # Cubic
        
        for j in range(n):
            if i != j:
                v = pos - state[j]
                dc = np.linalg.norm(v)
                ds = dc - 2*r
                if D_ROBOT > ds > EPS:
                    F += K_ROBOT * (1/ds - 1/D_ROBOT)**3 * v/dc  # Cubic
        
        do = np.linalg.norm(pos)
        dw = 1.0 - do
        if D_WALL > dw > EPS and do > EPS:
            F += K_WALL * (1/dw - 1/D_WALL)**3 * (-pos/do)  # Cubic
        
        velocities[i] = F
    return velocities


def v6_linear_log_repulsion(state, targets, obstacles, r):
    """Linear attraction with log repulsion (very soft)."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 6.0
    K_OBS, K_ROBOT, K_WALL = 0.03, 0.03, 0.002
    D_OBS, D_ROBOT, D_WALL = 0.2, 0.2, 0.01
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        F = K_ATT * (target - pos)
        
        for obs in obstacles:
            c = np.array(obs["center"])
            v = pos - c
            dc = np.linalg.norm(v)
            ds = dc - obs["radius"] - r
            if D_OBS > ds > EPS:
                F += K_OBS * (1/ds) * v/dc  # Log gradient
        
        for j in range(n):
            if i != j:
                v = pos - state[j]
                dc = np.linalg.norm(v)
                ds = dc - 2*r
                if D_ROBOT > ds > EPS:
                    F += K_ROBOT * (1/ds) * v/dc
        
        do = np.linalg.norm(pos)
        dw = 1.0 - do
        if D_WALL > dw > EPS and do > EPS:
            F += K_WALL * (1/dw) * (-pos/do)
        
        velocities[i] = F
    return velocities


def v7_quadratic_attraction(state, targets, obstacles, r):
    """Quadratic attraction (even stronger far away)."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 3.0
    K_OBS, K_ROBOT, K_WALL = 0.01, 0.01, 0.0002
    D_OBS, D_ROBOT, D_WALL = 0.15, 0.15, 0.005
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        vec = target - pos
        d = np.linalg.norm(vec)
        # Quadratic: force grows with distance squared (capped)
        F = K_ATT * min(d, 1.0) * vec
        
        for obs in obstacles:
            c = np.array(obs["center"])
            v = pos - c
            dc = np.linalg.norm(v)
            ds = dc - obs["radius"] - r
            if D_OBS > ds > EPS:
                F += K_OBS * (1/ds - 1/D_OBS)**2 * v/dc
        
        for j in range(n):
            if i != j:
                v = pos - state[j]
                dc = np.linalg.norm(v)
                ds = dc - 2*r
                if D_ROBOT > ds > EPS:
                    F += K_ROBOT * (1/ds - 1/D_ROBOT)**2 * v/dc
        
        do = np.linalg.norm(pos)
        dw = 1.0 - do
        if D_WALL > dw > EPS and do > EPS:
            F += K_WALL * (1/dw - 1/D_WALL)**2 * (-pos/do)
        
        velocities[i] = F
    return velocities


def v8_aggressive_linear(state, targets, obstacles, r):
    """Aggressive: very strong attraction, tight repulsion zones."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 12.0
    K_OBS, K_ROBOT, K_WALL = 0.02, 0.02, 0.0004
    D_OBS, D_ROBOT, D_WALL = 0.08, 0.08, 0.002
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        F = K_ATT * (target - pos)
        
        for obs in obstacles:
            c = np.array(obs["center"])
            v = pos - c
            dc = np.linalg.norm(v)
            ds = dc - obs["radius"] - r
            if D_OBS > ds > EPS:
                F += K_OBS * (1/ds - 1/D_OBS)**2 * v/dc
        
        for j in range(n):
            if i != j:
                v = pos - state[j]
                dc = np.linalg.norm(v)
                ds = dc - 2*r
                if D_ROBOT > ds > EPS:
                    F += K_ROBOT * (1/ds - 1/D_ROBOT)**2 * v/dc
        
        do = np.linalg.norm(pos)
        dw = 1.0 - do
        if D_WALL > dw > EPS and do > EPS:
            F += K_WALL * (1/dw - 1/D_WALL)**2 * (-pos/do)
        
        velocities[i] = F
    return velocities


if __name__ == "__main__":
    algorithms = [
        (v1_linear_base, "v1: Linear Base (K=5)"),
        (v2_stronger_linear, "v2: Stronger Linear (K=8)"),
        (v3_very_strong_linear, "v3: Very Strong Linear (K=10)"),
        (v4_hybrid_linear_const, "v4: Hybrid Linear-Const"),
        (v5_linear_cubic_repulsion, "v5: Linear + Cubic Repulsion"),
        (v6_linear_log_repulsion, "v6: Linear + Log Repulsion"),
        (v7_quadratic_attraction, "v7: Quadratic Attraction"),
        (v8_aggressive_linear, "v8: Aggressive (K=12)"),
    ]
    
    results = []
    for fn, name in algorithms:
        successes, avg_steps = test_algorithm(fn, name)
        results.append((name, successes, avg_steps))
    
    print("\n" + "="*60)
    print("ROUND 2 SUMMARY")
    print("="*60)
    results.sort(key=lambda x: (x[1] < 6, x[2]))
    for name, successes, avg_steps in results:
        steps_str = f"{avg_steps:.0f}" if avg_steps < float('inf') else "FAIL"
        status = "OK" if successes == 6 else "PARTIAL"
        print(f"{name:<35} {successes}/6 {steps_str:>8} steps")


