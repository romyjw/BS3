import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def load_obj(filename):
    vertices = []
    polyline = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:]])
            elif parts[0] == 'l':
                polyline.append([int(x) - 1 for x in parts[1:]])  # OBJ indices start at 1
    return np.array(vertices), polyline

def compute_winding_number(polyline, vertices, grid_x, grid_y):
    wn_grid = np.zeros_like(grid_x)
    
    for i in range(len(polyline) - 1):
        v1, v2 = vertices[polyline[i]], vertices[polyline[i+1]]
        
        dx1 = v1[0] - grid_x
        dy1 = v1[1] - grid_y
        dx2 = v2[0] - grid_x
        dy2 = v2[1] - grid_y
        
        angle1 = np.arctan2(dy1, dx1)
        angle2 = np.arctan2(dy2, dx2)
        
        dtheta = angle2 - angle1
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        
        wn_grid += dtheta / (2 * np.pi)
    
    return wn_grid

def plot_winding_number(grid_x, grid_y, wn_grid, vertices, polyline):
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(grid_x, grid_y, wn_grid, shading='auto', cmap='coolwarm', norm=Normalize(vmin=-1, vmax=1))
    plt.colorbar(label='Winding Number')
    
    for i in range(len(polyline) - 1):
        v1, v2 = vertices[polyline[i]], vertices[polyline[i+1]]
        plt.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-', lw=1)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Winding Number Visualization')
    plt.show()

# Load OBJ file
vertices, polyline = load_obj('meshes/eagle_bdry.obj')

# Create a grid
x_min, x_max = vertices[:, 0].min() - 1, vertices[:, 0].max() + 1
y_min, y_max = vertices[:, 1].min() - 1, vertices[:, 1].max() + 1
x_vals = np.linspace(x_min, x_max, 200)
y_vals = np.linspace(y_min, y_max, 200)
grid_x, grid_y = np.meshgrid(x_vals, y_vals)

# Compute winding number
wn_grid = compute_winding_number(polyline, vertices, grid_x, grid_y)

# Plot results
plot_winding_number(grid_x, grid_y, wn_grid, vertices, polyline)