import hakowan as hkw
import lagrange
import math


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
import os

SNSname = sys.argv[-1]


filepath = "../data/surfaces/fruit500.obj"


base =  hkw.layer(filepath).material(
    "Principled",
    color="#FBCD50",
    roughness=0.2,
    )
    
    
#wires = hkw.layer(filepath).mark("Curve").channel(size=0.0015).material("Diffuse", "black")
wires = hkw.layer(filepath).mark("Curve").channel(size=0.01).material("Diffuse", "black")

# Step 2: Adjust camera position.
config = hkw.config()
# Generate 4K rendering.
#config.film.width = 3840
#config.film.height = 2160


            
hkw.render(base+wires, config, filename="../rendering_results/wireframe-fruit.png")