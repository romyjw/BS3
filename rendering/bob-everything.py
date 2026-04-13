#!/usr/bin/env python

import math
import numpy as np
import matplotlib
import trimesh

import hakowan as hkw
import lagrange


# -----------------------------------------------------------------------------
# Filenames
# -----------------------------------------------------------------------------

proxy_filename = "bob500"

meancurv_inv_exp_surface_filename   = f"rendering_results/curvature/bob-inv-exp/meancurv.ply"
gausscurv_inv_exp_surface_filename   = f"rendering_results/curvature/bob-inv-exp/gausscurv.ply"

meancurv_trig_surface_filename   = f"rendering_results/curvature/bob-trig/meancurv.ply"
gausscurv_trig_surface_filename   = f"rendering_results/curvature/bob-trig/gausscurv.ply"

blended_surface_filename   = f"rendering_results/teaser/{proxy_filename}/normals.ply"

unblended_surface_filename = f"rendering_results/teaser/{proxy_filename}/unblended.ply"
mc_surface_filename = f"rendering_results/teaser/{proxy_filename}/mc.ply"

filepath = blended_surface_filename


# -----------------------------------------------------------------------------
# Shared rotation (DEFINE ONCE)
# -----------------------------------------------------------------------------

ROT_AXIS  = [0, 1, 0]
ROT_ANGLE = math.pi / 6

config = hkw.config()
config.sensor.location = [0, 2, 3]
config.integrator = hkw.setup.integrator.VolPath()


def rotated(layer):
    """Apply the global rotation once in a single place."""
    return layer.rotate(axis=ROT_AXIS, angle=ROT_ANGLE)


# -----------------------------------------------------------------------------
# Colour helpers
# -----------------------------------------------------------------------------

def mpl_discrete_colors(name, n=8):
    base = matplotlib.colormaps.get_cmap(name)
    colors = base(np.linspace(0, 1, n))[:, :3]
    return [tuple(c) for c in colors]


def random_rgb_colors(n):
    return [tuple(c) for c in np.random.rand(n, 3)]


# -----------------------------------------------------------------------------
# Load meshes
# -----------------------------------------------------------------------------

tm_mesh = trimesh.load(f"../data/surfaces/{proxy_filename}.ply")
n_vertices = tm_mesh.vertices.shape[0]

# Skeleton mesh
skeleton = lagrange.io.load_mesh(f"../data/surfaces/{proxy_filename}.ply")

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




########## mc layers #######################

 
mc_surface = lagrange.io.load_mesh(mc_surface_filename)

mc_base = rotated(
    hkw.layer(mc_surface).material(
        "Principled",
        "#F52749",  # pale red
        roughness=1.0,
        metallic=0.3,
    )
)

# Optional: add edges if you want
mc_edges = (
    mc_base
    .mark("Curve")
    .channel(size=0.002)
    .material("Principled", "black")
)

# Combine MC layers
mc_layers = mc_base + mc_edges





# -----------------------------------------------------------------------------
# Surface layers (ROTATION APPLIED ONCE)
# -----------------------------------------------------------------------------

surface_base = rotated(
    hkw.layer(filepath).material(
        "Principled",
        "#F54927",
        roughness=1.0,
        two_sided=True
    )
)

normals_layer = surface_base.material(
    "Principled",
    color=hkw.texture.ScalarField(
        data="normals_colours",
        colormap="identity",
    ),
)

unblended_layer = rotated(
    hkw.layer(unblended_surface_filename).material(
        "Principled",
        "#27A3F5",
        roughness=1.0,
        two_sided=True
    ),
)




meancurv_inv_exp = rotated(
    hkw.layer(meancurv_inv_exp_surface_filename).material(
    "Principled",
    color=hkw.texture.ScalarField(
        data="meancurv_colours",
        colormap="identity",
    ),
))

meancurv_trig = rotated(
    hkw.layer(meancurv_trig_surface_filename).material(
    "Principled",
    color=hkw.texture.ScalarField(
        data="meancurv_colours",
        colormap="identity",
    ),
))

gausscurv_inv_exp = rotated(
    hkw.layer(gausscurv_inv_exp_surface_filename).material(
    "Principled",
    color=hkw.texture.ScalarField(
        data="gausscurv_colours",
        colormap="identity",
    ),
))

gausscurv_trig = rotated(
    hkw.layer(gausscurv_trig_surface_filename).material(
    "Principled",
    color=hkw.texture.ScalarField(
        data="gausscurv_colours",
        colormap="identity",
    ),
))


# -----------------------------------------------------------------------------
# Render
# -----------------------------------------------------------------------------

curvature_layers = {
    'meancurv_inv_exp': meancurv_inv_exp,
    'gausscurv_inv_exp': gausscurv_inv_exp,
    'meancurv_trig': meancurv_trig,
    'gausscurv_trig': gausscurv_trig
}


for layerstring,layer in curvature_layers.items():
    hkw.render(
    layer,
    config,
    filename=f"rendering_results/main-results/{proxy_filename}/{layerstring}.png",
)










hkw.render(
    mc_layers,
    config,
    filename=f"rendering_results/teaser/{proxy_filename}/mc.png",
)


hkw.render(
    all_skeleton_layers,
    config,
    filename=f"rendering_results/teaser/{proxy_filename}/plain.png",
)



hkw.render(
    surface_base,
    config,
    filename=f"rendering_results/teaser/{proxy_filename}/one-colour.png",
)

hkw.render(
    normals_layer,
    config,
    filename=f"rendering_results/teaser/{proxy_filename}/normals.png",
)

hkw.render(
    unblended_layer ,
    config,
    filename=f"rendering_results/teaser/{proxy_filename}/unblended.png",
)
