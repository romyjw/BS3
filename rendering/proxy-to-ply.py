#!/usr/bin/env python

import lagrange
import numpy as np
import sys
from pathlib import Path


def obj_to_ply_with_vertex_labels(obj_path, ply_path):
    # Load OBJ
    mesh = lagrange.io.load_mesh(obj_path)

    # Number of vertices
    num_vertices = mesh.num_vertices

    # Create distinct labels: 0, 1, 2, ..., n-1
    #labels = np.arange(num_vertices, dtype=np.int32)
    labels = np.random.permutation(num_vertices).astype(np.int32)

    # Add vertex attribute
    mesh.create_attribute(
        "vertex_label",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=labels,
    )

    # Save as PLY (defaults to binary unless your build supports ascii options)
    lagrange.io.save_mesh(ply_path, mesh)

    print(f"Converted {obj_path} → {ply_path}")
    print(f"Added vertex_label attribute with {num_vertices} unique labels.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python proxy-to-ply.py input.obj output.ply")
        sys.exit(1)

    obj_path = Path(sys.argv[1])
    ply_path = Path(sys.argv[2])

    obj_to_ply_with_vertex_labels(obj_path, ply_path)
