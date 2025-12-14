import numpy as np
import math

def on_unit_circle(theta):
    return np.array([math.cos(theta), math.sin(theta)], dtype=float)

def inside_unit_disk(p):
    return np.linalg.norm(p) < 1.0 - 1e-12

def dist(a,b):
    return float(np.linalg.norm(a-b))

def violates_obstacle(p, obs, r):
    c = np.array(obs["center"], dtype=float)
    R = float(obs["radius"])
    return dist(p,c) <= R + r+ 1e-12

def any_robot_collision(state, r):
    n = state.shape[0]
    mind = float("inf")
    for i in range(n):
        for j in range(i+1, n):
            d = dist(state[i], state[j])
            if d < mind:
                mind = d
    return mind <= 2*r + 1e-12

def any_collision_with_obstacles(state, obstacles, r):
    for k in range(state.shape[0]):
        for obs in obstacles:
            if violates_obstacle(state[k], obs, r):
                return True
    return False

def any_outside_unit_disk(state):
    return np.any(np.linalg.norm(state, axis=1) > 1.0 + 1e-9)