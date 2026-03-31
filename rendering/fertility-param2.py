#!/usr/bin/env python

import math
import hakowan as hkw


# =============================================================================
# Configuration
# =============================================================================

PROXY_NAME = "fertility500"



# =============================================================================
# File paths
# =============================================================================

BASE_DIR = f"rendering_results/param/{PROXY_NAME}"

BLENDED_PATH   = f"{BASE_DIR}/param_bs.ply"
UNBLENDED_PATH = f"{BASE_DIR}/unblended.ply"

PROXY_PATH     = f"{BASE_DIR}/param_proxy.ply"


SKELETON_PATH  = f"../data/surfaces/{PROXY_NAME}.ply"


# =============================================================================
# Renderer setup
# =============================================================================

ROTATIONS = [
    ([0, 0, 1], -math.pi / 8),
    ([0, 1, 0], -math.pi / 6),
]

config = hkw.config()
config.sensor.location = [0, 0.5, 3]
config.integrator = hkw.setup.integrator.VolPath()


def rotated(layer):
    """Apply all global rotations in a single, consistent place."""
    for axis, angle in ROTATIONS:
        layer = layer.rotate(axis=axis, angle=angle)
    return layer

# =============================================================================
# Surface helper
# =============================================================================

def param_surface(filepath):
    return rotated(
        hkw.layer(filepath).material(
            "Principled",
            color=hkw.texture.ScalarField(
                data="param_colours",
                colormap="identity",
            ),
        )
    )


# =============================================================================
# Build layers
# =============================================================================

blended_surface   = param_surface(BLENDED_PATH)
unblended_surface = param_surface(UNBLENDED_PATH)
proxy_surface = param_surface(PROXY_PATH)




skeleton_base = rotated(
    hkw.layer(SKELETON_PATH).material(
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
        roughness=0.0,
        metallic=0.3,
    )
)



# =============================================================================
# Render
# =============================================================================

print("Rendering blended surface...")
hkw.render(
    blended_surface,
    config,
    filename=f"{BASE_DIR}/bs-param.png",
)

print("Rendering unblended surface...")
hkw.render(
    unblended_surface + skeleton_vertices,
    config,
    filename=f"{BASE_DIR}/unblended_bs-param.png",
)

print("Rendering proxy surface...")
hkw.render(
    proxy_surface + skeleton_vertices,
    config,
    filename=f"{BASE_DIR}/proxy-param.png",
)


'''
print("Rendering proxy surface...")
hkw.render(
    skeleton_edges + skeleton_vertices + skeleton_base,
    config,
    filename=f"{BASE_DIR}/skeleton.png",
)

'''