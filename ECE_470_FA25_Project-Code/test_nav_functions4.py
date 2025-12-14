"""
Round 4: Try fundamentally different approaches.
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


# Reference: Best so far is Linear Base with avg=8540 steps
def baseline_best(state, targets, obstacles, r):
    """Best from previous rounds - Linear K=5, D=0.15."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0
    K_OBS, K_ROBOT, K_WALL = 0.01, 0.01, 0.0002
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


def nav_function_kr(state, targets, obstacles, r):
    """Koditschek-Rimon style navigation function."""
    n = state.shape[0]
    EPS = 1e-12
    kappa = 2.0  # Tuning parameter
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        
        # Goal function: ||pos - target||^2
        gamma = np.sum((pos - target)**2)
        grad_gamma = 2 * (pos - target)
        
        # Barrier: product of obstacle distances
        beta = 1.0
        grad_beta = np.zeros(2)
        
        # Obstacle barriers
        for obs in obstacles:
            c = np.array(obs["center"])
            R = obs["radius"]
            d_sq = np.sum((pos - c)**2) - (R + r)**2
            if d_sq > EPS:
                beta *= d_sq
                grad_beta += 2 * (pos - c) / d_sq
        
        # Robot barriers
        for j in range(n):
            if i != j:
                d_sq = np.sum((pos - state[j])**2) - (2*r)**2
                if d_sq > EPS:
                    beta *= d_sq
                    grad_beta += 2 * (pos - state[j]) / d_sq
        
        # Boundary barrier: 1 - ||pos||^2
        boundary = 1.0 - np.sum(pos**2)
        if boundary > EPS:
            beta *= boundary
            grad_beta += -2 * pos / boundary
        
        # Navigation function: gamma / (gamma^kappa + beta)^(1/kappa)
        # Gradient computed via chain rule
        if beta > EPS and gamma > EPS:
            denom = (gamma**kappa + beta)**(1/kappa)
            
            # d/dx [gamma / denom]
            grad_nf = grad_gamma / denom - gamma / denom * (
                kappa * gamma**(kappa-1) * grad_gamma + grad_beta * beta
            ) / (kappa * (gamma**kappa + beta))
            
            velocities[i] = -5.0 * grad_nf  # Negative gradient with scaling
        else:
            velocities[i] = -grad_gamma
    
    return velocities


def adaptive_attraction(state, targets, obstacles, r):
    """Adaptive attraction - stronger when path is clear."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT_BASE = 5.0
    K_OBS, K_ROBOT, K_WALL = 0.01, 0.01, 0.0002
    D_OBS, D_ROBOT, D_WALL = 0.15, 0.15, 0.005
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        vec_to_goal = target - pos
        dist_to_goal = np.linalg.norm(vec_to_goal)
        
        # Check if path to goal is clear
        clearance = 1.0  # 1.0 = fully clear
        for obs in obstacles:
            c = np.array(obs["center"])
            dc = np.linalg.norm(pos - c)
            ds = dc - obs["radius"] - r
            if ds < D_OBS:
                clearance *= ds / D_OBS
        
        # Adaptive attraction: stronger when clear
        K_ATT = K_ATT_BASE * (0.5 + 0.5 * clearance)
        F = K_ATT * vec_to_goal
        
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


def priority_navigation(state, targets, obstacles, r):
    """Priority-based: robots with longer distances go first."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0
    K_OBS, K_ROBOT, K_WALL = 0.01, 0.01, 0.0002
    D_OBS, D_ROBOT, D_WALL = 0.15, 0.15, 0.005
    
    # Calculate distances to targets
    dists = np.array([np.linalg.norm(state[i] - targets[i]) for i in range(n)])
    priorities = np.argsort(-dists)  # Higher distance = higher priority
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        my_priority = np.where(priorities == i)[0][0]
        
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
                j_priority = np.where(priorities == j)[0][0]
                v = pos - state[j]
                dc = np.linalg.norm(v)
                ds = dc - 2*r
                if D_ROBOT > ds > EPS:
                    # Lower priority robots yield more
                    yield_factor = 1.5 if my_priority > j_priority else 0.5
                    F += yield_factor * K_ROBOT * (1/ds - 1/D_ROBOT)**2 * v/dc
        
        do = np.linalg.norm(pos)
        dw = 1.0 - do
        if D_WALL > dw > EPS and do > EPS:
            F += K_WALL * (1/dw - 1/D_WALL)**2 * (-pos/do)
        
        velocities[i] = F
    return velocities


def tangent_bug(state, targets, obstacles, r):
    """Tangent bug: follow obstacle boundary when blocked."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0
    K_OBS, K_ROBOT, K_WALL = 0.01, 0.01, 0.0002
    D_OBS, D_ROBOT, D_WALL = 0.15, 0.15, 0.005
    K_TANGENT = 0.5
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        vec_to_goal = target - pos
        goal_dir = vec_to_goal / (np.linalg.norm(vec_to_goal) + EPS)
        
        F = K_ATT * vec_to_goal
        
        for obs in obstacles:
            c = np.array(obs["center"])
            v = pos - c
            dc = np.linalg.norm(v)
            ds = dc - obs["radius"] - r
            
            if D_OBS > ds > EPS:
                radial = v / dc
                # Add tangent toward goal
                tangent = np.array([-radial[1], radial[0]])
                if np.dot(tangent, goal_dir) < 0:
                    tangent = -tangent
                
                # Blend radial and tangent
                blend = 1.0 - ds / D_OBS  # More tangent when closer
                F += K_OBS * (1/ds - 1/D_OBS)**2 * radial
                F += K_TANGENT * blend * (1/ds - 1/D_OBS) * tangent
        
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


def social_force(state, targets, obstacles, r):
    """Social force model - pedestrian-like navigation."""
    n = state.shape[0]
    EPS = 1e-12
    TAU = 0.5  # Relaxation time
    V0 = 1.0  # Desired speed
    A_OBS, B_OBS = 0.3, 0.1
    A_ROBOT, B_ROBOT = 0.2, 0.08
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        vec_to_goal = target - pos
        d_goal = np.linalg.norm(vec_to_goal)
        
        # Desired velocity
        if d_goal > EPS:
            v_desired = V0 * vec_to_goal / d_goal
        else:
            v_desired = np.zeros(2)
        
        # Driving force (toward goal)
        F = (v_desired - np.zeros(2)) / TAU  # Assuming current velocity is 0
        
        # Obstacle repulsion (social force)
        for obs in obstacles:
            c = np.array(obs["center"])
            v = pos - c
            dc = np.linalg.norm(v)
            ds = dc - obs["radius"] - r
            if ds > EPS:
                F += A_OBS * np.exp(-ds / B_OBS) * v / dc
        
        # Robot repulsion
        for j in range(n):
            if i != j:
                v = pos - state[j]
                dc = np.linalg.norm(v)
                ds = dc - 2*r
                if ds > EPS:
                    F += A_ROBOT * np.exp(-ds / B_ROBOT) * v / dc
        
        # Wall repulsion
        do = np.linalg.norm(pos)
        dw = 1.0 - do
        if dw > EPS and do > EPS:
            F += 0.1 * np.exp(-dw / 0.05) * (-pos / do)
        
        velocities[i] = F * 5.0  # Scale up
    return velocities


def higher_order_attraction(state, targets, obstacles, r):
    """Higher order attraction - accelerates near goal."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 4.0
    K_OBS, K_ROBOT, K_WALL = 0.01, 0.01, 0.0002
    D_OBS, D_ROBOT, D_WALL = 0.15, 0.15, 0.005
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        vec = target - pos
        d = np.linalg.norm(vec)
        
        # Higher order: F increases as we get closer (to speed up final approach)
        if d > 0.1:
            F = K_ATT * vec  # Linear far
        else:
            F = K_ATT * vec * (1.0 + 2.0 * (0.1 - d) / 0.1)  # Boost near
        
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
        (baseline_best, "Baseline (best so far)"),
        (nav_function_kr, "Koditschek-Rimon NF"),
        (adaptive_attraction, "Adaptive Attraction"),
        (priority_navigation, "Priority Navigation"),
        (tangent_bug, "Tangent Bug"),
        (social_force, "Social Force Model"),
        (higher_order_attraction, "Higher Order Attraction"),
    ]
    
    results = []
    for fn, name in algorithms:
        successes, avg_steps = test_algorithm(fn, name)
        results.append((name, successes, avg_steps))
    
    print("\n" + "="*60)
    print("ROUND 4 SUMMARY")
    print("="*60)
    results.sort(key=lambda x: (x[1] < 6, x[2]))
    for name, successes, avg_steps in results:
        steps_str = f"{avg_steps:.0f}" if avg_steps < float('inf') else "FAIL"
        print(f"{name:<35} {successes}/6 {steps_str:>8} steps")


