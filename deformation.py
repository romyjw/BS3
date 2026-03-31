import igl
import scipy as sp
import numpy as np
from meshplot import plot, subplot, interact

import os
root_folder = os.getcwd()




v, f = igl.read_triangle_mesh(os.path.join(root_folder, "data", "decimated-max.obj"))
v[:,[0, 2]] = v[:,[2, 0]] # Swap X and Z axes
u = v.copy()

s = igl.read_dmat(os.path.join(root_folder, "data", "decimated-max-selection.dmat"))
b = np.array([[t[0] for t in [(i, s[i]) for i in range(0, v.shape[0])] if t[1] >= 0]]).T

## Boundary conditions directly on deformed positions
u_bc = np.zeros((b.shape[0], v.shape[1]))
v_bc = np.zeros((b.shape[0], v.shape[1]))

for bi in range(b.shape[0]):
    v_bc[bi] = v[b[bi]]

    if s[b[bi]] == 0: # Don't move handle 0
        u_bc[bi] = v[b[bi]]
    elif s[b[bi]] == 1: # Move handle 1 down
        u_bc[bi] = v[b[bi]] + np.array([[0, -50, 0]])
    else: # Move other handles forward
        u_bc[bi] = v[b[bi]] + np.array([[-25, 0, 0]])

p = subplot(v, f, s, shading={"wireframe": False, "colormap": "tab10"}, s=[1, 4, 0])
for i in range(3):
    u_bc_anim = v_bc + i*0.6 * (u_bc - v_bc)
    d_bc = u_bc_anim - v_bc
    d = igl.harmonic(v, f, b, d_bc, 2)
    u = v + d
    subplot(u, f, s, shading={"wireframe": False, "colormap": "tab10"}, s=[1, 4, i+1], data=p)
p

# @interact(deformation_field=True, step=(0.0, 2.0))
# def update(deformation_field, step=0.0):
#     # Determine boundary conditions
#     u_bc_anim = v_bc + step * (u_bc - v_bc)

#     if deformation_field:
#         d_bc = u_bc_anim - v_bc
#         d = igl.harmonic(v, f, b, d_bc, 2)
#         u = v + d
#     else:
#         u = igl.harmonic(v, f, b, u_bc_anim, 2)
#     p.update_object(vertices=u)