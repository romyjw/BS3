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






for variation in ['trig', 'bary', 'inv-exp', 'simple-exp', 'disp-mlp']:
    identity = proxy_filename + '-' + variation
    

    blended_surface_filename   = f"rendering_results/interp/{identity}/normals.ply"
    filepath=blended_surface_filename
    
    # -----------------------------------------------------------------------------
    # Load meshes
    # -----------------------------------------------------------------------------
    
    tm_mesh = trimesh.load(f"../data/surfaces/{proxy_filename}.ply")
    n_vertices = tm_mesh.vertices.shape[0]
    
    # Skeleton mesh
    skeleton = lagrange.io.load_mesh(f"../data/surfaces/{proxy_filename}.ply")
    
    

    
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
        
    
    # -----------------------------------------------------------------------------
    # Render
    # -----------------------------------------------------------------------------
        
    
    hkw.render(
        surface_base,
        config,
        filename=f"rendering_results/interp/{identity}/one-colour.png",
    )
    
    hkw.render(
        normals_layer,
        config,
        filename=f"rendering_results/interp/{identity}/normals.png",
    )



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

    
    

        
    
