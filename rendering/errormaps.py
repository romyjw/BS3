#!/usr/bin/env python

import argparse
import math
import numpy as np
import trimesh

import hakowan as hkw
import lagrange


# =============================================================================
# Configuration
# =============================================================================

parser = argparse.ArgumentParser(description="Render / process a proxy mesh")
parser.add_argument(
    "proxy_name",
    type=str,
    help="Name of the proxy (e.g. twisted_torus, igea, bob, fertility)",
)
args = parser.parse_args()

PROXY_NAME = args.proxy_name


config = hkw.config()


if PROXY_NAME=='fertility' or PROXY_NAME=='fertility-djuren':
    ROTATIONS = [
        ([0, 0, 1], -math.pi / 8),
        ([0, 1, 0], -math.pi / 6),
    ]
    
    
    config.sensor.location = [0, 0.5, 3]
    
elif PROXY_NAME=='bob' or PROXY_NAME=='bob-djuren':
    ROTATIONS = [(  [0, 1, 0], math.pi / 6)]
    config.sensor.location = [0, 2, 3]
    
elif PROXY_NAME=='igea' or PROXY_NAME=='igea-djuren':
    ROTATIONS = [
    ([0, 1, 0], math.pi / 6)
    ]
    config.sensor.location = [0, 0.5, 3]
    
elif PROXY_NAME=='twisted_torus' or PROXY_NAME=='twisted_torus-djuren':
    ROTATIONS=[]
    config.sensor.location = [0, 0.5, 3]



def rotated(layer):
    """Apply all global rotations in a single, consistent place."""
    for axis, angle in ROTATIONS:
        layer = layer.rotate(axis=axis, angle=angle)
    return layer




filepath = f"rendering_results/errormaps/{PROXY_NAME}.ply"


bs = rotated( hkw.layer(filepath).material(
    "Principled",
    color=hkw.texture.ScalarField(
        data="error_colours",
        colormap="identity",
    ),
)
)



plain_layer = rotated(
    hkw.layer(filepath).material(
        "Principled",
        "#b366ff",
        roughness=1.0,
        two_sided=True
    ),
    
)









print("Rendering errormap...")
hkw.render(
    bs ,
    config,
    filename=f"rendering_results/errormaps/{PROXY_NAME}500.png",
)


print("Rendering plain version...")
hkw.render(
    plain_layer ,
    config,
    filename=f"rendering_results/errormaps/{PROXY_NAME}-plain-500.png",
)