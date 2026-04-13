#!/usr/bin/env python

import math
import numpy as np
import matplotlib
import trimesh

import hakowan as hkw
import lagrange




def mpl_discrete_colors(name, n=8):
    base = matplotlib.colormaps.get_cmap(name)
    colors = base(np.linspace(0, 1, n))[:, :3]
    return [tuple(c) for c in colors]


def random_rgb_colors(n):
    colors = [tuple(c) for c in np.random.rand(n, 3)]
    print(colors)
    return colors



n_vertices = 165
# -----------------------------------------------------------------------------
# Filenames
# -----------------------------------------------------------------------------


normals_surface_filename   = f"rendering_results/mobius/mobius-normals.ply"
abs_normals_surface_filename   = f"rendering_results/mobius/mobius-abs-normals.ply"
angle_surface_filename   = f"rendering_results/mobius/mobius-angle.ply"
coarse_angle_surface_filename   = f"rendering_results/mobius/mobius-coarse-angle.ply"
#skeleton_filename = "../data/surfaces/mobius224.ply"
skeleton_filename = "../data/surfaces/mobius-post-edit.ply"

# -----------------------------------------------------------------------------
# Shared rotation (DEFINE ONCE)
# -----------------------------------------------------------------------------

ROTATIONS = [
    ([1, 0, 0], math.pi / 2),
   ([0, 1, 0], -math.pi / 3)
]



config = hkw.config()
config.sensor.location = [0, 1.0, 3]
config.integrator = hkw.setup.integrator.VolPath()


def rotated(layer):
    """Apply all global rotations in a single, consistent place."""
    for axis, angle in ROTATIONS:
        layer = layer.rotate(axis=axis, angle=angle)
    return layer




skeleton_base = rotated(
    hkw.layer(skeleton_filename).material(
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
    .material("Conductor", "Cr", two_sided=True)
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












# -----------------------------------------------------------------------------
# Surface layers (ROTATION APPLIED ONCE)
# -----------------------------------------------------------------------------

hkw.render(
    skeleton_edges + skeleton_vertices + skeleton_base,
    config,
    filename=f"rendering_results/mobius/mobius-coarse.png"
   
)



normals_layer = rotated (hkw.layer(normals_surface_filename).material(
    "Principled",
    color=hkw.texture.ScalarField(
        data="normals_colours",
        colormap="identity",
    ),
    two_sided=True
)
)


plain_layer = rotated(
    hkw.layer(normals_surface_filename).material(
        "Principled",
        "#F54927",
        roughness=1.0,
        two_sided=True
    ),
    
)




hkw.render(
    plain_layer,
    config,
    filename=f"rendering_results/mobius/mobius-plain.png",
)




hkw.render(
    normals_layer,
    config,
    filename=f"rendering_results/mobius/mobius-normals.png",
)

abs_normals_layer = rotated (hkw.layer(abs_normals_surface_filename).material(
    "Principled",
    color=hkw.texture.ScalarField(
        data="abs-normals_colours",
        colormap="identity",
    ),
    two_sided=True
)
)


hkw.render(
    abs_normals_layer,
    config,
    filename=f"rendering_results/mobius/mobius-abs-normals.png",
)



angle_layer = rotated (hkw.layer(angle_surface_filename).material(
    "Principled",
    color=hkw.texture.ScalarField(
        data="angle-colours",
        colormap="identity",
    ),
    two_sided=True
)
)


hkw.render(
    angle_layer,
    config,
    filename=f"rendering_results/mobius/mobius-angle.png",
)




coarse_angle_layer = rotated (hkw.layer(coarse_angle_surface_filename).material(
    "Principled",
    color=hkw.texture.ScalarField(
        data="angle-colours",
        colormap="identity",
    ),
    two_sided=True
)
)


hkw.render(
    coarse_angle_layer + skeleton_edges,
    config,
    filename=f"rendering_results/mobius/mobius-coarse-angle.png",
)







