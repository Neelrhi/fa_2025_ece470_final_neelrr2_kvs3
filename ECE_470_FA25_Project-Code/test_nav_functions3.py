"""
Round 3: Focus on repulsion optimization and alternative approaches.
"""
import numpy as np
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


# Best so far: Linear K=5, D_OBS=0.15, D_ROBOT=0.15, D_WALL=0.005
# Try tighter influence zones and different formulas

def r1_tighter_zones(state, targets, obstacles, r):
    """Tighter repulsion zones - activate only when very close."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0
    K_OBS, K_ROBOT, K_WALL = 0.008, 0.008, 0.00015
    D_OBS, D_ROBOT, D_WALL = 0.1, 0.1, 0.004  # Tighter
    
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


def r2_very_tight_zones(state, targets, obstacles, r):
    """Very tight repulsion zones."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0
    K_OBS, K_ROBOT, K_WALL = 0.005, 0.005, 0.0001
    D_OBS, D_ROBOT, D_WALL = 0.08, 0.08, 0.003
    
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


def r3_linear_repulsion(state, targets, obstacles, r):
    """Linear (not inverse) repulsion - softer."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0
    K_OBS, K_ROBOT, K_WALL = 2.0, 2.0, 0.5
    D_OBS, D_ROBOT, D_WALL = 0.15, 0.15, 0.005
    
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
                # Linear repulsion: (D - ds) / D
                F += K_OBS * (D_OBS - ds) / D_OBS * v/dc
        
        for j in range(n):
            if i != j:
                v = pos - state[j]
                dc = np.linalg.norm(v)
                ds = dc - 2*r
                if D_ROBOT > ds > EPS:
                    F += K_ROBOT * (D_ROBOT - ds) / D_ROBOT * v/dc
        
        do = np.linalg.norm(pos)
        dw = 1.0 - do
        if D_WALL > dw > EPS and do > EPS:
            F += K_WALL * (D_WALL - dw) / D_WALL * (-pos/do)
        
        velocities[i] = F
    return velocities


def r4_sqrt_repulsion(state, targets, obstacles, r):
    """Square root repulsion - between linear and inverse."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0
    K_OBS, K_ROBOT, K_WALL = 0.1, 0.1, 0.01
    D_OBS, D_ROBOT, D_WALL = 0.15, 0.15, 0.005
    
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
                F += K_OBS * (1/np.sqrt(ds) - 1/np.sqrt(D_OBS)) * v/dc
        
        for j in range(n):
            if i != j:
                v = pos - state[j]
                dc = np.linalg.norm(v)
                ds = dc - 2*r
                if D_ROBOT > ds > EPS:
                    F += K_ROBOT * (1/np.sqrt(ds) - 1/np.sqrt(D_ROBOT)) * v/dc
        
        do = np.linalg.norm(pos)
        dw = 1.0 - do
        if D_WALL > dw > EPS and do > EPS:
            F += K_WALL * (1/np.sqrt(dw) - 1/np.sqrt(D_WALL)) * (-pos/do)
        
        velocities[i] = F
    return velocities


def r5_no_robot_repulsion(state, targets, obstacles, r):
    """No robot-robot repulsion - risky but potentially faster."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0
    K_OBS, K_WALL = 0.01, 0.0002
    D_OBS, D_WALL = 0.15, 0.005
    
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
        
        # NO robot repulsion
        
        do = np.linalg.norm(pos)
        dw = 1.0 - do
        if D_WALL > dw > EPS and do > EPS:
            F += K_WALL * (1/dw - 1/D_WALL)**2 * (-pos/do)
        
        velocities[i] = F
    return velocities


def r6_minimal_all(state, targets, obstacles, r):
    """Minimal everything - just enough to avoid collisions."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0
    K_OBS, K_ROBOT, K_WALL = 0.003, 0.003, 0.00005
    D_OBS, D_ROBOT, D_WALL = 0.08, 0.08, 0.003
    
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


def r7_smooth_blend(state, targets, obstacles, r):
    """Smooth polynomial blend for repulsion."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0
    K_OBS, K_ROBOT, K_WALL = 0.5, 0.5, 0.05
    D_OBS, D_ROBOT, D_WALL = 0.15, 0.15, 0.005
    
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
                # Smooth blend: (1 - ds/D)^2 * (2*ds/D + 1) / ds
                t = ds / D_OBS
                blend = (1 - t)**2 * (2*t + 1)
                F += K_OBS * blend / ds * v/dc
        
        for j in range(n):
            if i != j:
                v = pos - state[j]
                dc = np.linalg.norm(v)
                ds = dc - 2*r
                if D_ROBOT > ds > EPS:
                    t = ds / D_ROBOT
                    blend = (1 - t)**2 * (2*t + 1)
                    F += K_ROBOT * blend / ds * v/dc
        
        do = np.linalg.norm(pos)
        dw = 1.0 - do
        if D_WALL > dw > EPS and do > EPS:
            t = dw / D_WALL
            blend = (1 - t)**2 * (2*t + 1)
            F += K_WALL * blend / dw * (-pos/do)
        
        velocities[i] = F
    return velocities


def r8_optimized_baseline(state, targets, obstacles, r):
    """Optimized version of linear base - fine-tuned params."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0
    K_OBS, K_ROBOT, K_WALL = 0.007, 0.007, 0.00012
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


if __name__ == "__main__":
    algorithms = [
        (r1_tighter_zones, "r1: Tighter Zones (D=0.1)"),
        (r2_very_tight_zones, "r2: Very Tight (D=0.08)"),
        (r3_linear_repulsion, "r3: Linear Repulsion"),
        (r4_sqrt_repulsion, "r4: Sqrt Repulsion"),
        (r5_no_robot_repulsion, "r5: No Robot Repulsion"),
        (r6_minimal_all, "r6: Minimal All"),
        (r7_smooth_blend, "r7: Smooth Blend"),
        (r8_optimized_baseline, "r8: Optimized (D=0.12)"),
    ]
    
    results = []
    for fn, name in algorithms:
        successes, avg_steps = test_algorithm(fn, name)
        results.append((name, successes, avg_steps))
    
    print("\n" + "="*60)
    print("ROUND 3 SUMMARY")
    print("="*60)
    results.sort(key=lambda x: (x[1] < 6, x[2]))
    for name, successes, avg_steps in results:
        steps_str = f"{avg_steps:.0f}" if avg_steps < float('inf') else "FAIL"
        print(f"{name:<35} {successes}/6 {steps_str:>8} steps")


