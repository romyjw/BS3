#!/usr/bin/env python
"""
Render just the wobbly torus proxy (coarse control) meshes, at each of its
three resolutions -- the "plain" wireframe + rainbow-vertex-dot skeleton
view, same process as arm/gear/dice-everything.py (render_common's
rainbow vertex_label colouring + flat/Bilinear subdivision for the shaded
base), but with no BNS fine-mesh output involved at all.

Camera/rotation match the original rendering/wobbly-torus-remeshing.py
exactly: location [0, 2, 3], no rotation.

NOTE: proxy .ply files are generated into a dedicated folder under
rendering/rendering_results/wobbly_torus/, NOT into data/surfaces/ --
data/surfaces/wobbly_torus500.ply already exists there as an unrelated,
much denser mesh (the fitting target, not the proxy), so reusing that
naming convention would collide with it.
"""

import math

import hakowan as hkw

import render_common as rc


RESOLUTIONS = ["wobbly_torus200", "wobbly_torus300", "wobbly_torus500"]

OUTPUT_ROOT = rc.STABLE_RESULTS_ROOT / "wobbly_torus"

ROTATIONS = [([0, 1, 0], 0.0 * math.pi)]
rotate = rc.make_rotator(ROTATIONS)

config = hkw.config()
config.sensor.location = [0, 2, 3]
config.integrator = hkw.setup.integrator.VolPath()

for name in RESOLUTIONS:
    obj_path = rc.SURFACES_DIR / f"{name}.obj"

    mesh_dir = OUTPUT_ROOT / name
    mesh_dir.mkdir(parents=True, exist_ok=True)
    ply_path = mesh_dir / "proxy.ply"
    rc.ensure_ply_with_vertex_labels(obj_path, ply_path)

    skeleton, n_vertices = rc.build_skeleton(ply_path, obj_path)
    smooth_surface = rc.flat_subdivide(ply_path, num_levels=4)
    skeleton_layers = rc.build_skeleton_layers(skeleton, n_vertices, rotate, surface_mesh=smooth_surface)

    out_dir = mesh_dir / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "plain.png"
    hkw.render(skeleton_layers, config, filename=str(out_path))
    print(f"[wobbly_torus-proxies] rendered '{name}' -> {out_path}")
