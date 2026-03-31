#!/usr/bin/env python

import math
import numpy as np
import matplotlib
import trimesh

import hakowan as hkw
import lagrange





proxy_filename = "igea500"
degrees = [0,1,2,3]




ROTATIONS = [
    ([0, 1, 0], math.pi / 6)
]

config = hkw.config()
config.sensor.location = [0, 0.5, 3]
config.integrator = hkw.setup.integrator.VolPath()


def rotated(layer):
    """Apply all global rotations in a single, consistent place."""
    for axis, angle in ROTATIONS:
        layer = layer.rotate(axis=axis, angle=angle)
    return layer





for degree in range(4):
    filepath   = f"rendering_results/truncation/{proxy_filename}-degree{degree}.obj"
    
    
    
    surface_base = rotated(
        hkw.layer(filepath).material(
            "Principled",
            "#F54927",
            roughness=1.0,
            two_sided=True
        ),
        
    )
    
    
    # -----------------------------------------------------------------------------
    # Render
    # -----------------------------------------------------------------------------
        
    hkw.render(
        surface_base,
        config,
        filename=f"rendering_results/truncation/{proxy_filename}-degree{degree}.png",
    )
    
