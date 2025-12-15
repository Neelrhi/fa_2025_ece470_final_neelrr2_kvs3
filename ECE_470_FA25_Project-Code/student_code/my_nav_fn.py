import numpy as np

def compute_gradients(state, targets, obstacles, r):

    n = state.shape[0]
    epsilon_small = 1e-12
    
    goal_attraction = 5.0
    
    obstacle_repulsion = 0.01         
    robot_repulsion = 0.01      
    wall_repulsion = 0.0002
    
    obstacle_dist = 0.10         
    robot_dist = 0.10   
    wall_dist = 0.005     
    
    velocities = np.zeros_like(state)
    
    for i in range(n):
        pos = state[i]
        target = targets[i]
        
        F = goal_attraction * (target - pos)
        
        for obs in obstacles:
            c = np.array(obs["center"])
            v = pos - c 
            dc = np.linalg.norm(v)
            ds = dc - obs["radius"] - r
            
            if obstacle_dist > ds > epsilon_small:
                F += obstacle_repulsion * (1/ds - 1/obstacle_dist)**2 * v/dc
        
        for j in range(n):
            if i != j:
                v = pos - state[j] 
                dc = np.linalg.norm(v)
                ds = dc - 2*r 
                
                if robot_dist > ds > epsilon_small:
                    F += robot_repulsion * (1/ds - 1/robot_dist)**2 * v/dc
        
        dist_from_origin = np.linalg.norm(pos)
        dist_to_wall = 1.0 - dist_from_origin 
        
        if wall_dist > dist_to_wall > epsilon_small and dist_from_origin > epsilon_small:
            F += wall_repulsion * (1/dist_to_wall - 1/wall_dist)**2 * (-pos/dist_from_origin)
        
        velocities[i] = F
    
    return velocities
