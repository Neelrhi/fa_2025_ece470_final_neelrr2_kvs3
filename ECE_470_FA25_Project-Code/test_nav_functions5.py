"""
Round 5: Iterate on Tangent Bug - the new best performer.
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


def t1_tangent_base(state, targets, obstacles, r):
    """Base Tangent Bug from round 4."""
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
                tangent = np.array([-radial[1], radial[0]])
                if np.dot(tangent, goal_dir) < 0:
                    tangent = -tangent
                
                blend = 1.0 - ds / D_OBS
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


def t2_stronger_tangent(state, targets, obstacles, r):
    """Stronger tangent component."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0
    K_OBS, K_ROBOT, K_WALL = 0.01, 0.01, 0.0002
    D_OBS, D_ROBOT, D_WALL = 0.15, 0.15, 0.005
    K_TANGENT = 0.8  # Increased
    
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
                tangent = np.array([-radial[1], radial[0]])
                if np.dot(tangent, goal_dir) < 0:
                    tangent = -tangent
                
                blend = 1.0 - ds / D_OBS
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


def t3_tangent_everywhere(state, targets, obstacles, r):
    """Apply tangent to robot-robot interactions too."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0
    K_OBS, K_ROBOT, K_WALL = 0.01, 0.008, 0.0002
    D_OBS, D_ROBOT, D_WALL = 0.15, 0.15, 0.005
    K_TANGENT_OBS = 0.5
    K_TANGENT_ROBOT = 0.3
    
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
                tangent = np.array([-radial[1], radial[0]])
                if np.dot(tangent, goal_dir) < 0:
                    tangent = -tangent
                
                blend = 1.0 - ds / D_OBS
                F += K_OBS * (1/ds - 1/D_OBS)**2 * radial
                F += K_TANGENT_OBS * blend * (1/ds - 1/D_OBS) * tangent
        
        for j in range(n):
            if i != j:
                v = pos - state[j]
                dc = np.linalg.norm(v)
                ds = dc - 2*r
                if D_ROBOT > ds > EPS:
                    radial = v / dc
                    tangent = np.array([-radial[1], radial[0]])
                    if np.dot(tangent, goal_dir) < 0:
                        tangent = -tangent
                    
                    blend = 1.0 - ds / D_ROBOT
                    F += K_ROBOT * (1/ds - 1/D_ROBOT)**2 * radial
                    F += K_TANGENT_ROBOT * blend * (1/ds - 1/D_ROBOT) * tangent
        
        do = np.linalg.norm(pos)
        dw = 1.0 - do
        if D_WALL > dw > EPS and do > EPS:
            F += K_WALL * (1/dw - 1/D_WALL)**2 * (-pos/do)
        
        velocities[i] = F
    return velocities


def t4_tangent_tighter(state, targets, obstacles, r):
    """Tangent with tighter influence zones."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0
    K_OBS, K_ROBOT, K_WALL = 0.008, 0.008, 0.00015
    D_OBS, D_ROBOT, D_WALL = 0.12, 0.12, 0.004
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
                tangent = np.array([-radial[1], radial[0]])
                if np.dot(tangent, goal_dir) < 0:
                    tangent = -tangent
                
                blend = 1.0 - ds / D_OBS
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


def t5_tangent_plus_priority(state, targets, obstacles, r):
    """Combine tangent bug with priority navigation."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0
    K_OBS, K_ROBOT, K_WALL = 0.01, 0.01, 0.0002
    D_OBS, D_ROBOT, D_WALL = 0.15, 0.15, 0.005
    K_TANGENT = 0.5
    
    # Calculate priorities
    dists = np.array([np.linalg.norm(state[i] - targets[i]) for i in range(n)])
    priorities = np.argsort(-dists)
    
    velocities = np.zeros_like(state)
    for i in range(n):
        pos, target = state[i], targets[i]
        vec_to_goal = target - pos
        goal_dir = vec_to_goal / (np.linalg.norm(vec_to_goal) + EPS)
        my_priority = np.where(priorities == i)[0][0]
        
        F = K_ATT * vec_to_goal
        
        for obs in obstacles:
            c = np.array(obs["center"])
            v = pos - c
            dc = np.linalg.norm(v)
            ds = dc - obs["radius"] - r
            
            if D_OBS > ds > EPS:
                radial = v / dc
                tangent = np.array([-radial[1], radial[0]])
                if np.dot(tangent, goal_dir) < 0:
                    tangent = -tangent
                
                blend = 1.0 - ds / D_OBS
                F += K_OBS * (1/ds - 1/D_OBS)**2 * radial
                F += K_TANGENT * blend * (1/ds - 1/D_OBS) * tangent
        
        for j in range(n):
            if i != j:
                j_priority = np.where(priorities == j)[0][0]
                v = pos - state[j]
                dc = np.linalg.norm(v)
                ds = dc - 2*r
                if D_ROBOT > ds > EPS:
                    yield_factor = 1.3 if my_priority > j_priority else 0.7
                    F += yield_factor * K_ROBOT * (1/ds - 1/D_ROBOT)**2 * v/dc
        
        do = np.linalg.norm(pos)
        dw = 1.0 - do
        if D_WALL > dw > EPS and do > EPS:
            F += K_WALL * (1/dw - 1/D_WALL)**2 * (-pos/do)
        
        velocities[i] = F
    return velocities


def t6_adaptive_tangent(state, targets, obstacles, r):
    """Tangent strength adapts to obstacle proximity."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0
    K_OBS, K_ROBOT, K_WALL = 0.01, 0.01, 0.0002
    D_OBS, D_ROBOT, D_WALL = 0.15, 0.15, 0.005
    
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
                tangent = np.array([-radial[1], radial[0]])
                if np.dot(tangent, goal_dir) < 0:
                    tangent = -tangent
                
                # Adaptive tangent: stronger when very close
                blend = (1.0 - ds / D_OBS) ** 2  # Quadratic blend
                K_TANGENT = 0.3 + 0.7 * blend  # 0.3 to 1.0
                
                F += K_OBS * (1/ds - 1/D_OBS)**2 * radial
                F += K_TANGENT * (1/ds - 1/D_OBS) * tangent
        
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


def t7_full_tangent(state, targets, obstacles, r):
    """Full tangent: stronger tangent, less radial."""
    n = state.shape[0]
    EPS = 1e-12
    K_ATT = 5.0
    K_OBS, K_ROBOT, K_WALL = 0.008, 0.01, 0.0002
    D_OBS, D_ROBOT, D_WALL = 0.15, 0.15, 0.005
    K_TANGENT = 1.0  # Full tangent strength
    
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
                tangent = np.array([-radial[1], radial[0]])
                if np.dot(tangent, goal_dir) < 0:
                    tangent = -tangent
                
                blend = 1.0 - ds / D_OBS
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


if __name__ == "__main__":
    algorithms = [
        (t1_tangent_base, "t1: Tangent Base"),
        (t2_stronger_tangent, "t2: Stronger Tangent (K=0.8)"),
        (t3_tangent_everywhere, "t3: Tangent Everywhere"),
        (t4_tangent_tighter, "t4: Tangent Tighter (D=0.12)"),
        (t5_tangent_plus_priority, "t5: Tangent + Priority"),
        (t6_adaptive_tangent, "t6: Adaptive Tangent"),
        (t7_full_tangent, "t7: Full Tangent (K=1.0)"),
    ]
    
    results = []
    for fn, name in algorithms:
        successes, avg_steps = test_algorithm(fn, name)
        results.append((name, successes, avg_steps))
    
    print("\n" + "="*60)
    print("ROUND 5 SUMMARY - TANGENT BUG VARIANTS")
    print("="*60)
    results.sort(key=lambda x: (x[1] < 6, x[2]))
    for name, successes, avg_steps in results:
        steps_str = f"{avg_steps:.0f}" if avg_steps < float('inf') else "FAIL"
        print(f"{name:<35} {successes}/6 {steps_str:>8} steps")


