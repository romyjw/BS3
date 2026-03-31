#!/usr/bin/env python

import hakowan as hkw
import lagrange
import numpy as np
import math
import matplotlib
import numpy as np
import trimesh


proxy_filename = "bob500"
blended_surface_filename = 'rendering_results/teaser/normals_coloured.ply'




def mpl_discrete_colors(name, n=8):
    """
    Return n discrete RGB colors from a named matplotlib colormap.
    Compatible with Matplotlib >= 3.7
    """
    base = matplotlib.colormaps.get_cmap(name)
    colors = base(np.linspace(0, 1, n))[:, :3]  # drop alpha
    return [tuple(c) for c in colors]


def random_rgb_colors(n):
    """
    Return n random RGB colors in [0,1] as tuples.
    """
    return [tuple(c) for c in np.random.rand(n, 3)]
    
    
tm_mesh = trimesh.load("../data/surfaces/" + proxy_filename + ".ply")
n_vertices = tm_mesh.vertices.shape[0]

# Step 1: Load skeleton.
skeleton = lagrange.io.load_mesh("../data/surfaces/" + proxy_filename + ".ply")
with open("../data/surfaces/" + proxy_filename + ".obj", "r") as fin:
    for line in fin:
        if line.startswith("l "):
            fields = line.split()
            skeleton.add_polygon(np.array([int(fields[1]) - 1, int(fields[2]) - 1]))


# Step 2: Load base mesh with glass like material.
#base = hkw.layer("../data/surfaces/deep3d_fertility-mc.ply").material("ThinDielectric")

skeleton_base = hkw.layer(skeleton).material("Conductor", "Cr")
#skeleton_base = hkw.layer(skeleton).material("Principled", "black", roughness=0, metallic=0.8)


skeleton_edges = skeleton_base.mark("Curve").channel(size=0.005)






vertices = (
    skeleton_base.mark("Point")
    .channel(size=0.015)
    .material(
        "Principled",
        hkw.texture.ScalarField(
            "vertex_label",
            colormap = random_rgb_colors(n_vertices),
            #colormap=mpl_discrete_colors("tab20", n=500)
            #colormap=["steelblue", "green", "yellow", "red", "orange", "purple"]
        ),
        roughness=0,
        metallic=0.3,
    )
)


#vertex_bubbles = (
#    skeleton_base.mark("Point")
#    .channel(size=0.2)
#    .material("ThinDielectric")

#)


# Step 3: Combine all layers
all_layers = (   skeleton_edges + vertices + skeleton_base ).rotate(
    axis=[0, 1, 0], angle=math.pi / 6
)


# Step 4: Adjust camera and render.
config = hkw.config()
config.sensor.location = [0, 2, 3]
config.integrator = hkw.setup.integrator.VolPath()
hkw.render(
    all_layers,
    config,
    filename="rendering_results/" + proxy_filename + ".png",
)






# Create layer
base = hkw.layer(filepath).material(
    "Principled",
    "#8080FF",
    roughness=0.8,
).rotate(
    axis=[0, 1, 0], angle=math.pi / 6
)


normals_layer = base.material("Principled", color=hkw.texture.ScalarField(
       data="normals_colours", colormap='identity'
    )  )

config = hkw.config()
config.sensor.location = [0, 2, 3]
#config.albedo_only = True

hkw.render(base, config, filename="rendering_results/teaser/one-colour-bob.png")
hkw.render(normals_layer, config, filename="rendering_results/teaser/normals-bob.png")



