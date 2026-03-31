import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from IPython.display import clear_output
import time

# Parameters
Du, Dv = 0.16, 0.08
F, k = 0.037, 0.06
dt = 0.01
steps = 30000
update_interval = 100

# Create icosphere mesh
mesh = trimesh.creation.icosphere(subdivisions=5)
vertices = mesh.vertices
faces = mesh.faces
n = len(vertices)

# Laplacian matrix
L = trimesh.smoothing.laplacian_calculation(mesh, equal_weight=True)

# Normalize Laplacian
from scipy.sparse import diags
def normalize_laplacian(L):
    d = np.abs(L).sum(axis=1).A1
    D_inv = diags(1 / d)
    return D_inv @ L

L = normalize_laplacian(L)

# Initialize fields
U = np.ones(n)
V = np.zeros(n)

# Seed V near a point
dist = np.linalg.norm(vertices - vertices[0], axis=1)
V[dist < 0.1] = np.exp(-dist[dist < 0.1]**2 )
#V[dist < 0.5] = np.exp(-dist[dist < 0.5]**2 )
#V[dist > 0.9] = np.exp(-dist[dist > 0.9]**2 )
U -= V

# Set up plot
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection='3d')
ax.set_axis_off()

# Function to update mesh colors based on V values
def update_plot(ax, V):
    #ax.collections.clear()
    V_clipped = np.clip(V, 0, 1)
    cmap = plt.get_cmap('inferno')
    colors = cmap(V_clipped)

    poly3d = [[vertices[i] for i in face] for face in faces]
    collection = Poly3DCollection(poly3d, facecolors=colors[faces].mean(axis=1), linewidths=0.1)
    ax.add_collection3d(collection)
    ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])

# Initial display
update_plot(ax, V)
plt.show(block=False)

# Simulation loop
for i in range(steps):
    #print(i)
    lap_U = L @ U
    lap_V = L @ V

    UVV = U * V * V
    U += dt * (Du * lap_U - UVV + F * (1 - U))
    V += dt * (Dv * lap_V + UVV - (F + k) * V)

    if i % update_interval == 0:
        update_plot(ax, V)
        ax.set_title(f"Step {i}", pad=20)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.01)
