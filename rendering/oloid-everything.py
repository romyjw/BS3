#!/usr/bin/env python
"""
Render the arm shape's BNS output (normals / unblended / one-colour / ...)
using Hakowan.

Reads its input .ply/.obj files from rendering/rendering_results/arm/, the
stable folder that BPS_visualiser.show_bps(shape_name="arm") writes into
(run the notebook's visualisation cell for the arm config first) — no
manual file copying needed.

To turn a layer's render on/off, just flip its value in ENABLED_SETTINGS
below.
"""

import math

import hakowan as hkw

import render_common as rc


SHAPE_NAME = "oloid"
PROXY_FILENAME = "oloid100"  # matches surface_config['coarse_patches_id']

INPUT_DIR = rc.STABLE_RESULTS_ROOT / SHAPE_NAME
OUTPUT_DIR = INPUT_DIR / "renders"

# Per-shape camera tuning — copied from bob-everything.py as a starting
# point, retune by eye once you've seen the first render.
ROTATIONS = [([0, 1, 0], math.pi / 6), ([0, 0, 1], math.pi / 2), ([1, 0, 0], math.pi * 1.8 )]
rotate = rc.make_rotator(ROTATIONS)

config = hkw.config()
config.sensor.location = [0, 0, 3]
config.integrator = hkw.setup.integrator.VolPath()

proxy_ply, proxy_obj = rc.resolve_proxy_paths(INPUT_DIR, PROXY_FILENAME)
skeleton, n_vertices = rc.build_skeleton(proxy_ply, proxy_obj)
skeleton_layers = rc.build_skeleton_layers(skeleton, n_vertices, rotate)
registry = rc.make_setting_registry(INPUT_DIR, rotate)

# --- Toggle which layers get rendered (edit here, nothing else needed) ---
ENABLED_SETTINGS = {
    'plain':       True,
    'one-colour':  True,
    'normals':     True,
    'unblended':   True,
    'abs-normals': False,
    'meancurv':    False,
    'gausscurv':   False,
    'error':True
}

rc.render_toggled(ENABLED_SETTINGS, registry, skeleton_layers, config, OUTPUT_DIR)
