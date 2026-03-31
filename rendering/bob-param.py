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

bs_filepath       = f"rendering_results/param/{proxy_filename}/param_bs.ply"
proxy_filepath    = f"rendering_results/param/{proxy_filename}/param_proxy.ply"
skeleton_filepath = f"../data/surfaces/{proxy_filename}.ply"


# -----------------------------------------------------------------------------
# Shared rotation (DEFINE ONCE)
# -----------------------------------------------------------------------------

ROT_AXIS  = [0, 1, 0]
ROT_ANGLE = math.pi / 6

config = hkw.config()
config.sensor.location = [0, 2, 3]
config.integrator = hkw.setup.integrator.VolPath()


def rotated(layer):
    """Apply the global rotation once."""
    return layer.rotate(axis=ROT_AXIS, angle=ROT_ANGLE)


# -----------------------------------------------------------------------------
# Colour helpers
# -----------------------------------------------------------------------------

def random_rgb_colors(n):
    return [tuple(c) for c in np.random.rand(n, 3)]


# -----------------------------------------------------------------------------
# Load skeleton
# -----------------------------------------------------------------------------

tm_mesh = trimesh.load(skeleton_filepath)
n_vertices = tm_mesh.vertices.shape[0]

skeleton = lagrange.io.load_mesh(skeleton_filepath)

# Load OBJ line segments as curves
#with open(skeleton_filepath, "r") as fin:
#    for line in fin:
#        if line.startswith("l "):
#            _, i, j = line.split()
#            skeleton.add_polygon(np.array([int(i) - 1, int(j) - 1]))


# -----------------------------------------------------------------------------
# Skeleton layers
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
            "vertex_label",                     # <-- vertex attribute
            colormap=random_rgb_colors(n_vertices),
        ),
        roughness=0.0,
        metallic=0.3,
    )
)




# -----------------------------------------------------------------------------
# Surface layers (param_colours)
# -----------------------------------------------------------------------------

bs_base = rotated( hkw.layer(bs_filepath)
        .material(
        "Principled",
        color = hkw.texture.ScalarField(
        data="param_colours",
        colormap="identity",
    )
)
)

proxy_base = rotated(
hkw.layer(proxy_filepath)
    .material(
        "Principled",
        color = hkw.texture.ScalarField(
        data="param_colours",
        colormap="identity",
    )
))


# -----------------------------------------------------------------------------
# Render
# -----------------------------------------------------------------------------

hkw.render(
    bs_base + skeleton_vertices,
    config,
    filename=f"rendering_results/param/{proxy_filename}/bs-param.png",
)

hkw.render(
    proxy_base + skeleton_vertices + skeleton_edges,
    config,
    filename=f"rendering_results/param/{proxy_filename}/proxy-param.png",
)
