#!/usr/bin/env python

import math
import numpy as np
import trimesh

import hakowan as hkw
import lagrange


# =============================================================================
# Configurationinit
# =============================================================================

PROXY_NAME = "fertility"

ROTATIONS = [
    ([0, 0, 1], -math.pi / 8),
    ([0, 1, 0], -math.pi / 6),
]



filepath = f"rendering_results/errormaps/{PROXY_NAME}.ply"


bs = hkw.layer(filepath)

# =============================================================================
# Renderer setup
# =============================================================================

config = hkw.config()
config.sensor.location = [0, 0.5, 3]
# config.integrator = hkw.setup.integrator.VolPath()


def rotated(layer):
    """Apply all global rotations in a single, consistent place."""
    for axis, angle in ROTATIONS:
        layer = layer.rotate(axis=axis, angle=angle)
    return layer



print("Rendering BS...")
hkw.render(
    bs ,
    config,
    filename=f"{BASE_DIR}/bs-param.png",
)
