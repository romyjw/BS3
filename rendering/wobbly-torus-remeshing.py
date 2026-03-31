#!/usr/bin/env python

import math
import numpy as np
import matplotlib
import trimesh

import hakowan as hkw
import lagrange
import subprocess


# -----------------------------------------------------------------------------
# Filenames
# -----------------------------------------------------------------------------

proxy_filenames = [
    "wobbly_torus500",
    "wobbly_torus200",
    "wobbly_torus300",
    "wobbly_torus500-diffmesh",
]


# -----------------------------------------------------------------------------
# Shared rotation (DEFINE ONCE)
# -----------------------------------------------------------------------------

ROT_AXIS  = [0, 1, 0]
ROT_ANGLE = 0.0 * math.pi


def rotated(layer):
    """Apply the global rotation once in a single place."""
    return layer.rotate(axis=ROT_AXIS, angle=ROT_ANGLE)


# -----------------------------------------------------------------------------
# Renderer config
# -----------------------------------------------------------------------------

config = hkw.config()
config.sensor.location = [0, 2, 3]
config.integrator = hkw.setup.integrator.VolPath()


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
# Surface layer builders
# -----------------------------------------------------------------------------

def make_surface_layers(filepath):
    """Return (plain_surface, normals_surface)."""

    surface_base = rotated(
        hkw.layer(filepath).material(
            "Principled",
            "#F54927",
            roughness=1.0,
            two_sided=True,
        )
    )

    normals_layer = surface_base.material(
        "Principled",
        color=hkw.texture.ScalarField(
            data="normals_colours",
            colormap="identity",
        ),
    )

    return surface_base, normals_layer


# -----------------------------------------------------------------------------
# Skeleton layer builders
# -----------------------------------------------------------------------------

def make_skeleton_layers(skeleton, n_vertices):
    """Return combined skeleton layers."""

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

    return skeleton_base + skeleton_edges + skeleton_vertices


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

for proxy_filename in proxy_filenames:

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------

    obj_path          = f"../data/surfaces/{proxy_filename}.obj"
    coarse_ply_path   = f"../data/surfaces/{proxy_filename}_coarse.ply"
    subdiv_ply_path   = f"../data/surfaces/{proxy_filename}_subdiv.ply"

    # -------------------------------------------------------------------------
    # Convert OBJ → coarse PLY (NO subdivision)
    # -------------------------------------------------------------------------

    subprocess.run(
        ["python", "proxy-to-ply.py", obj_path, coarse_ply_path],
        check=True
    )

    # -------------------------------------------------------------------------
    # Load coarse mesh (for skeleton)
    # -------------------------------------------------------------------------

    coarse_tm = trimesh.load(coarse_ply_path)
    n_vertices = coarse_tm.vertices.shape[0]

    skeleton = lagrange.io.load_mesh(coarse_ply_path)

    # -------------------------------------------------------------------------
    # Build subdivided mesh (for surface rendering only)
    # -------------------------------------------------------------------------

    subdiv_tm = coarse_tm
    for _ in range(4):
        subdiv_tm = subdiv_tm.subdivide()

    subdiv_tm.export(subdiv_ply_path)

    # -------------------------------------------------------------------------
    # Build layers
    # -------------------------------------------------------------------------

    surface_ply = f"rendering_results/remeshing/normals-{proxy_filename}.ply"

    surface_plain, surface_normals = make_surface_layers(surface_ply)
    skeleton_layers = make_skeleton_layers(skeleton, n_vertices)

    # -------------------------------------------------------------------------
    # Render passes (SEPARATE SOURCES)
    # -------------------------------------------------------------------------

    # Plain subdivided surface
    hkw.render(
        surface_plain,
        config,
        filename=f"rendering_results/remeshing/{proxy_filename}-plain.png",
    )

    # Normals colouring (subdivided)
    hkw.render(
        surface_normals,
        config,
        filename=f"rendering_results/remeshing/{proxy_filename}-normals.png",
    )

    # Skeleton (coarse, non-subdivided)
    hkw.render(
        skeleton_layers,
        config,
        filename=f"rendering_results/remeshing/{proxy_filename}-skeleton.png",
    )
