#!/usr/bin/env python

import hakowan as hkw
import lagrange
import numpy as np
import math


proxy_filename = "bob500"

# Step 1: Load skeleton.
skeleton = lagrange.io.load_mesh("../data/surfaces/" + proxy_filename + ".obj")
with open("../data/surfaces/" + proxy_filename + ".obj", "r") as fin:
    for line in fin:
        if line.startswith("l "):
            fields = line.split()
            skeleton.add_polygon(np.array([int(fields[1]) - 1, int(fields[2]) - 1]))


# Step 2: Load base mesh with glass like material.
#base = hkw.layer("../data/surfaces/deep3d_fertility-mc.ply").material("ThinDielectric")

skeleton_base = hkw.layer(skeleton).material("Conductor", "Cr")
skeleton_edges = skeleton_base.mark("Curve").channel(size=0.01)


vertices = (
    skeleton_base.mark("Point")
    .channel(size=0.03)
    .material(
        "Conductor", "Cr"
        )
)


# Step 3: Combine all layers
all_layers = (   skeleton_edges + vertices  ).rotate(
    axis=[0, 1, 0], angle=math.pi / 6
)

# Step 4: Adjust camera and render.
config = hkw.config()
config.sensor.location = [0, 2, 3]
config.integrator = hkw.setup.integrator.VolPath()
hkw.render(
    all_layers,
    config,
    filename="rendering_results/" + proxy_filename + ".png",
)