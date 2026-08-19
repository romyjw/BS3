#!/usr/bin/env python
"""
Shared helpers for the per-shape "<shape>-everything.py" Hakowan rendering
scripts (arm/gear/dice, and future shapes following the same pattern).

Reads colored .ply/.obj files from the stable per-shape folder that
BPS_visualiser.show_bps() now writes to (rendering/rendering_results/<shape>/),
and turns them into Hakowan layers. Which layers actually get rendered is
controlled by a plain {setting_name: bool} dict passed to render_toggled(),
so enabling/disabling a setting is a one-line change in the driver script
instead of commenting code in/out.
"""

from pathlib import Path

import numpy as np
import matplotlib
import lagrange
import hakowan as hkw


REPO_ROOT = Path(__file__).resolve().parent.parent
SURFACES_DIR = REPO_ROOT / "data" / "surfaces"
STABLE_RESULTS_ROOT = Path(__file__).resolve().parent / "rendering_results"


def ensure_ply_with_vertex_labels(obj_path, ply_path):
    """
    Make sure ply_path exists and is newer than obj_path, (re)generating it
    if needed (same approach as rendering/proxy-to-ply.py: a lagrange-loaded
    mesh with a random per-vertex "vertex_label" attribute, which the
    skeleton-vertices layer colours by). This is the only place in the
    whole pipeline that touches lagrange for the proxy mesh -- kept
    confined to this rendering script's environment, as before.

    Checking mtime (not just existence) matters because show_bps()
    overwrites obj_path fresh every run -- without it, a .ply cached from a
    previous run of a different proxy mesh (e.g. after redoing a shape's
    coarse mesh) would silently keep being reused, mismatched with the
    current .obj.
    """
    obj_path = Path(obj_path)
    ply_path = Path(ply_path)
    stale = ply_path.exists() and obj_path.stat().st_mtime > ply_path.stat().st_mtime

    if not ply_path.exists() or stale:
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        mesh = lagrange.io.load_mesh(str(obj_path))
        labels = np.random.permutation(mesh.num_vertices).astype(np.int32)
        mesh.create_attribute(
            "vertex_label",
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=labels,
        )
        lagrange.io.save_mesh(str(ply_path), mesh)
        print(f"[render_common] generated {ply_path}")

    return ply_path


def mpl_discrete_colors(name, n=8):
    base = matplotlib.colormaps.get_cmap(name)
    colors = base(np.linspace(0, 1, n))[:, :3]
    return [tuple(c) for c in colors]


def random_rgb_colors(n):
    return [tuple(c) for c in np.random.rand(n, 3)]


def make_rotator(rotations):
    """rotations: list of (axis, angle) tuples, applied in order to a layer."""

    def rotate(layer):
        for axis, angle in rotations:
            layer = layer.rotate(axis=axis, angle=angle)
        return layer

    return rotate


def resolve_proxy_paths(shape_dir, proxy_filename):
    """
    Find the proxy .obj -- preferring the copy BPS_visualiser.show_bps() now
    writes into shape_dir (rendering/rendering_results/<shape_name>/proxy.obj),
    falling back to data/surfaces/<proxy_filename>.obj (the pre-existing
    convention) if the shape folder doesn't have one yet -- then generate a
    matching "vertex_label" .ply alongside it if one isn't already cached
    there. Same lagrange-based conversion as before, just pointed at
    whichever .obj we found.
    """
    shape_dir = Path(shape_dir)
    obj_path = shape_dir / "proxy.obj"
    if not obj_path.exists():
        obj_path = SURFACES_DIR / f"{proxy_filename}.obj"

    ply_path = obj_path.with_suffix(".ply")
    ensure_ply_with_vertex_labels(obj_path, ply_path)
    return ply_path, obj_path


def build_skeleton(ply_path, obj_path):
    """
    Load a proxy mesh's .ply (with its "vertex_label" attribute) plus its
    .obj's "l " polyline lines, and return (skeleton_mesh, n_vertices).
    """
    skeleton = lagrange.io.load_mesh(str(ply_path))
    n_vertices = skeleton.num_vertices

    with open(obj_path, "r") as fin:
        for line in fin:
            if line.startswith("l "):
                _, i, j = line.split()
                skeleton.add_polygon(np.array([int(i) - 1, int(j) - 1]))

    return skeleton, n_vertices


def flat_subdivide(ply_path, num_levels=2):
    """
    Load ply_path and subdivide it with the Bilinear scheme -- a purely
    topological (flat) split of each face, no Catmull-Clark/Loop smoothing,
    so the mesh's shape/bounding box is unchanged. Used only to make the
    skeleton's shaded base surface look less faceted -- a single
    degenerate/dark triangle in the coarse proxy gets broken into many
    small ones instead of showing up as one large dark patch.
    """
    mesh = lagrange.io.load_mesh(str(ply_path))
    if num_levels > 0:
        mesh = lagrange.subdivision.subdivide_mesh(
            mesh, num_levels, scheme=lagrange.subdivision.SchemeType.Bilinear,
        )
    return mesh


def build_skeleton_layers(skeleton, n_vertices, rotate, surface_mesh=None):
    """
    The wireframe + coloured-vertex-dots "skeleton" overlay, plus a shaded
    base surface. surface_mesh (e.g. from flat_subdivide()) is used for the
    shaded surface only, if given -- the wireframe edges and vertex_label
    dots always come from `skeleton` itself, since they rely on its
    original vertex indices/attribute.
    """
    wire_base = rotate(
        hkw.layer(skeleton).material(
            "Principled", "#E9ECF2", roughness=1.0, metallic=0.3, two_sided=True,
        )
    )

    skeleton_edges = (
        wire_base
        .mark("Curve")
        .channel(size=0.005)
        .material("Conductor", "Cr")
    )

    skeleton_vertices = (
        wire_base
        .mark("Point")
        .channel(size=0.015)
        .material(
            "Principled",
            hkw.texture.ScalarField(
                "vertex_label",
                colormap=random_rgb_colors(n_vertices),
                # categories=True: vertex_label is a raw integer id (0..n-1),
                # not an already-normalised [0,1] value. Without this,
                # ColorMap clamps every label >1 to the same end colour, so
                # every vertex ends up the same hue instead of a rainbow.
                categories=True,
            ),
            roughness=0,
            metallic=0.3,
        )
    )

    skeleton_base = rotate(
        hkw.layer(surface_mesh if surface_mesh is not None else skeleton).material(
            "Principled", "#E9ECF2", roughness=1.0, metallic=0.3, two_sided=True,
        )
    )

    return skeleton_base + skeleton_edges + skeleton_vertices


def make_setting_registry(shape_dir, rotate):
    """
    shape_dir: rendering/rendering_results/<shape_name>/ — the stable folder
    BPS_visualiser.show_bps() writes into.

    Returns {setting_name: () -> (hkw_layer, png_filename)}. Each entry is
    self-contained (reads only its own .ply, no cross-setting dependency).
    """
    shape_dir = Path(shape_dir)

    def scalar_layer(ply_name, field, roughness=1.0, two_sided=True):
        # NB: hakowan's .material() calls don't merge -- the last one in the
        # chain fully replaces the layer's material channel. So two_sided
        # has to live on this final call (the one with the ScalarField
        # colour), not on an earlier one, or it's silently discarded.
        base = rotate(hkw.layer(str(shape_dir / ply_name)))
        return base.material(
            "Principled",
            color=hkw.texture.ScalarField(data=field, colormap="identity"),
            roughness=roughness,
            two_sided=two_sided,
        )

    def flat_layer(ply_name, flat_hex, two_sided=True):
        return rotate(
            hkw.layer(str(shape_dir / ply_name)).material(
                "Principled", flat_hex, roughness=1.0, two_sided=two_sided,
            )
        )

    return {
        'one-colour':  lambda: (flat_layer("uniform-coloured.ply", "#F54927"), "one-colour.png"),
        'unblended':   lambda: (flat_layer("unblended.ply", "#27A3F5"), "unblended.png"),
        'normals':     lambda: (scalar_layer("normals.ply", "normals_colours"), "normals.png"),
        'abs-normals': lambda: (scalar_layer("abs-normals.ply", "abs-normals_colours"), "abs-normals.png"),
        'meancurv':    lambda: (scalar_layer("meancurv.ply", "meancurv_colours"), "meancurv.png"),
        'gausscurv':   lambda: (scalar_layer("gausscurv.ply", "gausscurv_colours"), "gausscurv.png"),
        'error':       lambda: (scalar_layer("error.ply", "error_colours"), "error.png"),
    }


def render_toggled(enabled, registry, skeleton_layers, config, out_dir):
    """
    enabled: {setting_name: bool} (or a set/list of enabled names).
    'plain' renders skeleton_layers; every other enabled name is looked up
    in `registry`. Writes one PNG per enabled setting into out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = enabled if isinstance(enabled, (set, list)) else {k for k, v in enabled.items() if v}

    for name in sorted(names):
        if name == 'plain':
            layer, png_name = skeleton_layers, "plain.png"
        elif name in registry:
            layer, png_name = registry[name]()
        else:
            raise KeyError(f"Unknown render setting '{name}'. Available: 'plain', {sorted(registry)}")

        out_path = out_dir / png_name
        hkw.render(layer, config, filename=str(out_path))
        print(f"[render_common] rendered '{name}' -> {out_path}")
