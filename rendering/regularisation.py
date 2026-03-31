#!/usr/bin/env python

import argparse
import math
import hakowan as hkw

# =============================================================================
# Configuration
# =============================================================================

parser = argparse.ArgumentParser(description="Render / process a proxy mesh")
parser.add_argument(
    "proxy_name",
    type=str,
    help="proxy mesh name",
)
parser.add_argument(
    "reg_type",
    type=str,
    help="which version? vanilla, nreg, or nreg+?",
)
args = parser.parse_args()

PROXY_NAME = args.proxy_name
REG_TYPE   = args.reg_type

config = hkw.config()



# Generate 4K rendering.
config.film.width = 3840
config.film.height = 2160



ROTATIONS = [
    ([0, 0, 1], -math.pi / 8),
    ([0, 1, 0], -math.pi / 6),
]

config.sensor.location = [0, 0.5, 3]


def rotated(layer):
    """Apply all global rotations in a single, consistent place."""
    for axis, angle in ROTATIONS:
        layer = layer.rotate(axis=axis, angle=angle)
    return layer


# =============================================================================
# File
# =============================================================================

filepath = f"rendering_results/regularisation/{PROXY_NAME}-normals-{REG_TYPE}.ply"


# =============================================================================
# 1. Normals render
# =============================================================================

bs_normals = rotated(
    hkw.layer(filepath).material(
        "Principled",
        color=hkw.texture.ScalarField(
            data="normals_colours",
            colormap="identity",
        ),
    )
)

bs_plain = rotated(
    hkw.layer(filepath).material(
        "Principled",
        color="#F54927",
        roughness=1.0,
        metallic=0.0,
    )
)



bs_edges = bs_plain.mark("Curve").channel(size=0.0005).material("Principled",color= "#000000")


print("Rendering normals...")
hkw.render(
    bs_normals,
    config,
    filename=f"rendering_results/regularisation/{PROXY_NAME}-{REG_TYPE}-normals.png",
)


# =============================================================================
# 2. Plain coloured mesh
# =============================================================================
print("Rendering plain mesh with edges...")
hkw.render(
    bs_edges+bs_plain,
    config,
    filename=f"rendering_results/regularisation/{PROXY_NAME}-{REG_TYPE}-plain-edges.png",
)





print("Rendering plain mesh...")
hkw.render(
    bs_plain,
    config,
    filename=f"rendering_results/regularisation/{PROXY_NAME}-{REG_TYPE}-plain.png",
)



