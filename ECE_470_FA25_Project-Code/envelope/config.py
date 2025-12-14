# config.py
CONFIG = {
    "n": 5,
    "r": 0.05,
    "dt": 0.005,
    "vmax": 0.05,
    "tol": 0.005,
    "max_steps": 30000,
    "show_progress": True,    # print/update progress while running
    "progress_every": 5,  
}

OBSTACLES = [
    {"center": (0.0, 0.0), "radius": 0.1},
    {"center": (0.5, 0.2), "radius": 0.1},
    {"center": (-0.3, -0.3), "radius": 0.1},
]
