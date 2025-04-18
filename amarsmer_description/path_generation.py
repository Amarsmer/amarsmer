import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import pathlib

# Path parameters
radius = 2.0        # meters
depth_per_circle = 1.0  # meters
num_turns = 3
steps_per_turn = 100
total_steps = num_turns * steps_per_turn
z_start = 0.0

# Generate the helical path
t_vals = np.linspace(0, 2 * np.pi * num_turns, total_steps)
x_vals = radius * np.cos(t_vals)
y_vals = radius * np.sin(t_vals)
z_vals = z_start - (depth_per_circle * t_vals) / (2 * np.pi)

# Orientation: yaw follows path tangent
poses = []

for i in range(total_steps):
    x, y, z = x_vals[i], y_vals[i], z_vals[i]
    
    # Tangent vector to get yaw
    dx = -radius * np.sin(t_vals[i])
    dy = radius * np.cos(t_vals[i])
    yaw = np.arctan2(dy, dx)  # heading direction
    
    # Convert to quaternion (roll=0, pitch=0)
    quat = R.from_euler('zyx', [yaw, 0, 0]).as_quat()  # [x, y, z, w]
    
    pose = {
        'position': (x, y, z),
        'orientation': {
            'x': quat[0],
            'y': quat[1],
            'z': quat[2],
            'w': quat[3],
        }
    }
    poses.append(pose)

current_path = str(pathlib.Path(__file__).parent.resolve())

# Save poses to JSON so it can be reloaded later for SDF generation
output_path = current_path + '/poses.json'
with open(output_path, "w") as f:
    json.dump(poses, f, indent=2)
