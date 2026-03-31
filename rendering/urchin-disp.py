#!/usr/bin/env python

import math
import numpy as np
import hakowan as hkw
import lagrange
import trimesh

# -----------------------------------------------------------------------------
# Filenames
# -----------------------------------------------------------------------------
proxy_filename = "urchin210"

blended_surface_filename = f"rendering_results/disp/{proxy_filename}/bs-normals.ply"
disp_surface_filename    = f"rendering_results/disp/{proxy_filename}/disp.obj"
mc_surface_filename    = f"rendering_results/disp/{proxy_filename}/mc9.ply"

skeleton_filename_obj = f"../data/surfaces/{proxy_filename}.obj"
skeleton_filename_ply = f"../data/surfaces/{proxy_filename}.ply"



import trimesh

tm = trimesh.load(mc_surface_filename)

for _ in range(4):
    tm = tm.subdivide()
tm.export(mc_surface_filename)
    
    
import subprocess

subprocess.run(
    ["python", "proxy-to-ply.py", skeleton_filename_obj, skeleton_filename_ply],
    check=True
)



# -----------------------------------------------------------------------------
# Shared rotation
# -----------------------------------------------------------------------------
ROT_AXIS  = [0, 1, 0]
ROT_ANGLE = 0.0

config = hkw.config()
config.sensor.location = [0, -2, 3]
config.integrator = hkw.setup.integrator.VolPath()

def rotated(layer):
    """Apply the global rotation once."""
    return layer.rotate(axis=ROT_AXIS, angle=ROT_ANGLE)

# -----------------------------------------------------------------------------
# Helper: discrete colors (if needed)
# -----------------------------------------------------------------------------
def random_rgb_colors(n):
    return [tuple(c) for c in np.random.rand(n, 3)]




# -----------------------------------------------------------------------------
# Load meshes
# -----------------------------------------------------------------------------

tm_mesh = trimesh.load(skeleton_filename_obj)
n_vertices = tm_mesh.vertices.shape[0]

# Skeleton mesh
skeleton = lagrange.io.load_mesh(skeleton_filename_ply)

with open(f"../data/surfaces/{proxy_filename}.obj", "r") as fin:
    for line in fin:
        if line.startswith("l "):
            _, i, j = line.split()
            skeleton.add_polygon(np.array([int(i) - 1, int(j) - 1]))


# -----------------------------------------------------------------------------
# Skeleton layers (ROTATION APPLIED ONCE)
# -----------------------------------------------------------------------------

skeleton_base = rotated(
    hkw.layer(skeleton).material(
        "Principled",
        "#E9ECF2",
        roughness=1.0,
        metallic=0.3,
        two_sided=True
        
    )
)

skeleton_edges = (
    skeleton_base
    .mark("Curve")
    .channel(size=0.005)
    .material("Conductor", "Cr")
)

skeleton_vertices = (
    skeleton_base
    .mark("Point")
    .channel(size=0.015)
    .material(
        "Principled",
        hkw.texture.ScalarField(
            "vertex_label",
            colormap=random_rgb_colors(n_vertices),
        ),
        roughness=0,
        metallic=0.3,
    )
)

all_skeleton_layers = (
    skeleton_base
    + skeleton_edges
    + skeleton_vertices
)










# -----------------------------------------------------------------------------
# Layers
# -----------------------------------------------------------------------------
disp_layer = rotated(
    hkw.layer(disp_surface_filename).material(
        "Principled",
        "#de2323",  # red-ish
        roughness=0.3,
        two_sided=True
    )
)


mc_layer = rotated(
    hkw.layer(mc_surface_filename).material(
        "Principled",
        "#4263cf",  # blue
        roughness=0.3,
        two_sided=True
        
    ),
    
)







'''
blended_layer = rotated(
    hkw.layer(blended_surface_filename).material(
        "Principled",
        color=hkw.texture.ScalarField(
            data="normals_colours",  # use vertex normals as colors
            colormap="identity"
        ),
        roughness=1.0,
        two_sided=True
    )
)
'''

blended_layer = rotated(
    hkw.layer(blended_surface_filename).material(
        "Principled",
        "#F54927",
        roughness=0.3,
        two_sided=True
    )
)


# Generate 4K rendering.
config.film.width = 3840
config.film.height = 2160


# -----------------------------------------------------------------------------
# Render
# -----------------------------------------------------------------------------


hkw.render(
    mc_layer,
    config,
    filename=f"rendering_results/disp/{proxy_filename}/mc.png"
)

hkw.render(
    blended_layer,
    config,
    filename=f"rendering_results/disp/{proxy_filename}/blended.png"
)

hkw.render(
    all_skeleton_layers,
    config,
    filename=f"rendering_results/disp/{proxy_filename}/skeleton.png"
)


hkw.render(
    disp_layer,
    config,
    filename=f"rendering_results/disp/{proxy_filename}/disp.png"
)