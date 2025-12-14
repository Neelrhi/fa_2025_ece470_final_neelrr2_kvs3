"""
Test different navigation function approaches to find the fastest one.
"""
import numpy as np
import time
from envelope.simulate import simulate_once, targets_from_permutation
from envelope.config import OBSTACLES

# Test permutations (diverse sample)
TEST_PERMS = [
    (0,1,2,3,4), (4,3,2,1,0), (2,0,4,1,3), 
    (1,3,0,4,2), (3,4,1,2,0), (0,2,4,1,3)
]

def test_algorithm(compute_fn, name, max_steps=30000):
    """Test an algorithm on sample permutations."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)
    
    successes = 0
    total_steps = 0
    start_time = time.time()
    
    for sigma in TEST_PERMS:
        result = simulate_once(compute_fn, n=5, sigma=sigma, r=0.05, 
                               dt=0.005, vmax=0.05, tol=0.005, max_steps=max_steps)
        success = result.get('success')
        steps = result.get('steps')
        reason = result.get('reason')
        
        status = "OK" if success else "FAIL"
        print(f"  {status} Perm {sigma}: {steps} steps ({reason})")
        
        if success:
            successes += 1
            total_steps += steps
    
    elapsed = time.time() - start_time
    success_rate = successes / len(TEST_PERMS) * 100
    avg_steps = total_steps / successes if successes > 0 else float('inf')
    
    print(f"\nResults: {successes}/{len(TEST_PERMS)} success ({success_rate:.0f}%)")
    print(f"Average steps (successful): {avg_steps:.0f}")
    print(f"Compute time: {elapsed:.1f}s")
    
    return successes, avg_steps, elapsed


# ============================================================================
# ALGORITHM 1: Current APF (Baseline)
# Linear attraction, inverse-square repulsion
# ============================================================================
def alg1_baseline(state, targets, obstacles, r):
    """Baseline APF: constant magnitude attraction, inverse-square repulsion."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT, K_OBS, K_ROBOT, K_WALL = 1.0, 0.005, 0.005, 0.0001
    D_OBS, D_ROBOT, D_WALL = 0.2, 0.2, 0.005
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        
        # Attraction
        vec = target - pos
        d = np.linalg.norm(vec)
        F = (K_ATT * vec / d) if d > EPS else np.zeros(2)
        
        # Obstacle repulsion
        for obs in obstacles:
            c = np.array(obs["center"])
            v = pos - c
            dc = np.linalg.norm(v)
            ds = dc - obs["radius"] - r
            if D_OBS > ds > EPS:
                F += K_OBS * (1/ds - 1/D_OBS)**2 * v/dc
        
        # Robot repulsion
        for j in range(n):
            if i != j:
                v = pos - state[j]
                dc = np.linalg.norm(v)
                ds = dc - 2*r
                if D_ROBOT > ds > EPS:
                    F += K_ROBOT * (1/ds - 1/D_ROBOT)**2 * v/dc
        
        # Wall repulsion
        do = np.linalg.norm(pos)
        dw = 1.0 - do
        if D_WALL > dw > EPS and do > EPS:
            F += K_WALL * (1/dw - 1/D_WALL)**2 * (-pos/do)
        
        velocities[i] = F
    return velocities


# ============================================================================
# ALGORITHM 2: Stronger Attraction
# Much higher attraction, slightly higher repulsion
# ============================================================================
def alg2_strong_attraction(state, targets, obstacles, r):
    """Stronger attraction for faster convergence."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT, K_OBS, K_ROBOT, K_WALL = 3.0, 0.01, 0.01, 0.0002
    D_OBS, D_ROBOT, D_WALL = 0.15, 0.15, 0.003
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        vec = target - pos
        d = np.linalg.norm(vec)
        F = (K_ATT * vec / d) if d > EPS else np.zeros(2)
        
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


# ============================================================================
# ALGORITHM 3: Logarithmic Barriers
# Log barrier functions for smoother gradients
# ============================================================================
def alg3_log_barriers(state, targets, obstacles, r):
    """Logarithmic barrier functions."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 2.0
    K_OBS, K_ROBOT, K_WALL = 0.02, 0.02, 0.001
    D_OBS, D_ROBOT, D_WALL = 0.2, 0.2, 0.01
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        vec = target - pos
        d = np.linalg.norm(vec)
        F = (K_ATT * vec / d) if d > EPS else np.zeros(2)
        
        for obs in obstacles:
            c = np.array(obs["center"])
            v = pos - c
            dc = np.linalg.norm(v)
            ds = dc - obs["radius"] - r
            if D_OBS > ds > EPS:
                # Log barrier: -d/dx log(ds) = -1/ds * gradient
                F += K_OBS * (1/ds) * v/dc
        
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


# ============================================================================
# ALGORITHM 4: Exponential Repulsion
# Exponential decay for repulsion (smoother far field)
# ============================================================================
def alg4_exponential(state, targets, obstacles, r):
    """Exponential repulsion functions."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 2.0
    K_OBS, K_ROBOT, K_WALL = 0.5, 0.5, 0.1
    SIGMA_OBS, SIGMA_ROBOT, SIGMA_WALL = 0.05, 0.05, 0.01
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        vec = target - pos
        d = np.linalg.norm(vec)
        F = (K_ATT * vec / d) if d > EPS else np.zeros(2)
        
        for obs in obstacles:
            c = np.array(obs["center"])
            v = pos - c
            dc = np.linalg.norm(v)
            ds = dc - obs["radius"] - r
            if ds > EPS and ds < 0.3:
                # Exponential: exp(-ds/sigma) / sigma
                F += K_OBS * np.exp(-ds/SIGMA_OBS) / SIGMA_OBS * v/dc
        
        for j in range(n):
            if i != j:
                v = pos - state[j]
                dc = np.linalg.norm(v)
                ds = dc - 2*r
                if ds > EPS and ds < 0.3:
                    F += K_ROBOT * np.exp(-ds/SIGMA_ROBOT) / SIGMA_ROBOT * v/dc
        
        do = np.linalg.norm(pos)
        dw = 1.0 - do
        if dw > EPS and dw < 0.05 and do > EPS:
            F += K_WALL * np.exp(-dw/SIGMA_WALL) / SIGMA_WALL * (-pos/do)
        
        velocities[i] = F
    return velocities


# ============================================================================
# ALGORITHM 5: Capped Linear Attraction
# Linear attraction (proportional to distance) with cap
# ============================================================================
def alg5_linear_attraction(state, targets, obstacles, r):
    """Linear attraction proportional to distance."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0  # Will be scaled by distance
    K_OBS, K_ROBOT, K_WALL = 0.01, 0.01, 0.0002
    D_OBS, D_ROBOT, D_WALL = 0.15, 0.15, 0.005
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        vec = target - pos
        d = np.linalg.norm(vec)
        # Linear: force proportional to distance (spring-like)
        F = K_ATT * vec  # Not normalized - magnitude grows with distance
        
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


# ============================================================================
# ALGORITHM 6: Vectorized APF
# Same as baseline but fully vectorized with NumPy
# ============================================================================
def alg6_vectorized(state, targets, obstacles, r):
    """Fully vectorized APF for speed."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT, K_OBS, K_ROBOT, K_WALL = 1.0, 0.005, 0.005, 0.0001
    D_OBS, D_ROBOT, D_WALL = 0.2, 0.2, 0.005
    
    # Attraction (vectorized)
    vec_to_goal = targets - state
    dist_to_goal = np.linalg.norm(vec_to_goal, axis=1, keepdims=True)
    F_att = K_ATT * vec_to_goal / (dist_to_goal + EPS)
    
    # Obstacle repulsion (vectorized over robots, loop over obstacles)
    F_obs = np.zeros_like(state)
    for obs in obstacles:
        c = np.array(obs["center"])
        vec = state - c
        dc = np.linalg.norm(vec, axis=1, keepdims=True)
        ds = dc - obs["radius"] - r
        mask = (ds < D_OBS) & (ds > EPS)
        strength = np.where(mask, K_OBS * (1/ds - 1/D_OBS)**2, 0)
        F_obs += strength * vec / (dc + EPS)
    
    # Robot-robot repulsion (need pairwise)
    F_robot = np.zeros_like(state)
    for i in range(n):
        for j in range(n):
            if i != j:
                v = state[i] - state[j]
                dc = np.linalg.norm(v)
                ds = dc - 2*r
                if D_ROBOT > ds > EPS:
                    F_robot[i] += K_ROBOT * (1/ds - 1/D_ROBOT)**2 * v/dc
    
    # Wall repulsion (vectorized)
    dist_origin = np.linalg.norm(state, axis=1, keepdims=True)
    dist_wall = 1.0 - dist_origin
    mask = (dist_wall < D_WALL) & (dist_wall > EPS) & (dist_origin > EPS)
    strength = np.where(mask, K_WALL * (1/dist_wall - 1/D_WALL)**2, 0)
    F_wall = strength * (-state) / (dist_origin + EPS)
    
    return F_att + F_obs + F_robot + F_wall


# ============================================================================
# ALGORITHM 7: Minimal Repulsion
# Bare minimum repulsion for maximum speed
# ============================================================================
def alg7_minimal_repulsion(state, targets, obstacles, r):
    """Minimal repulsion - only activate when very close."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 1.0
    K_OBS, K_ROBOT, K_WALL = 0.002, 0.002, 0.00005
    D_OBS, D_ROBOT, D_WALL = 0.1, 0.1, 0.003
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        vec = target - pos
        d = np.linalg.norm(vec)
        F = (K_ATT * vec / d) if d > EPS else np.zeros(2)
        
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


# ============================================================================
# ALGORITHM 8: Hybrid with Tangent
# Add tangential component near obstacles for faster navigation
# ============================================================================
def alg8_with_tangent(state, targets, obstacles, r):
    """Add tangential force near obstacles to navigate around faster."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 1.5
    K_OBS, K_ROBOT, K_WALL = 0.005, 0.005, 0.0001
    K_TANGENT = 0.3
    D_OBS, D_ROBOT, D_WALL = 0.2, 0.2, 0.005
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        vec = target - pos
        d = np.linalg.norm(vec)
        goal_dir = vec / d if d > EPS else np.zeros(2)
        F = K_ATT * goal_dir
        
        for obs in obstacles:
            c = np.array(obs["center"])
            v = pos - c
            dc = np.linalg.norm(v)
            ds = dc - obs["radius"] - r
            if D_OBS > ds > EPS:
                radial = v / dc
                # Add tangential component
                tangent_cw = np.array([-radial[1], radial[0]])
                tangent_ccw = np.array([radial[1], -radial[0]])
                # Pick tangent toward goal
                tangent = tangent_cw if np.dot(tangent_cw, goal_dir) > np.dot(tangent_ccw, goal_dir) else tangent_ccw
                
                strength = (1/ds - 1/D_OBS)**2
                F += K_OBS * strength * radial + K_TANGENT * strength * tangent
        
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


# ============================================================================
# RUN ALL TESTS
# ============================================================================
if __name__ == "__main__":
    algorithms = [
        (alg1_baseline, "1. Baseline APF"),
        (alg2_strong_attraction, "2. Strong Attraction"),
        (alg3_log_barriers, "3. Logarithmic Barriers"),
        (alg4_exponential, "4. Exponential Repulsion"),
        (alg5_linear_attraction, "5. Linear Attraction"),
        (alg6_vectorized, "6. Vectorized APF"),
        (alg7_minimal_repulsion, "7. Minimal Repulsion"),
        (alg8_with_tangent, "8. With Tangent"),
    ]
    
    results = []
    for fn, name in algorithms:
        successes, avg_steps, elapsed = test_algorithm(fn, name)
        results.append((name, successes, avg_steps, elapsed))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - Sorted by Average Steps (lower is better)")
    print("="*70)
    print(f"{'Algorithm':<30} {'Success':>10} {'Avg Steps':>12} {'Time':>10}")
    print("-"*70)
    
    # Sort by avg_steps (successful ones first, then by steps)
    results.sort(key=lambda x: (x[1] < len(TEST_PERMS), x[2]))
    
    for name, successes, avg_steps, elapsed in results:
        success_str = f"{successes}/{len(TEST_PERMS)}"
        steps_str = f"{avg_steps:.0f}" if avg_steps < float('inf') else "N/A"
        print(f"{name:<30} {success_str:>10} {steps_str:>12} {elapsed:>8.1f}s")

