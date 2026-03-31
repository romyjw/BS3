import numpy as np
import open3d as o3d
import igl
from pathlib import Path

if __name__ == "__main__":
    mesh = o3d.io.read_triangle_mesh('data/surfaces/bob.obj')
    v = mesh.vertices
    f = mesh.triangles

    # v = np.asarray(v) / 1.5

    q_rand_points = np.random.uniform(-1.0, 1.0, (200000, 3))
    q_surface_points = mesh.sample_points_uniformly(200000)
    q_surface_points = np.asarray(q_surface_points.points)
    q_surface_points = [
        q_surface_points + np.random.normal(scale=0.001, size=(len(q_surface_points), 3)),
        q_surface_points + np.random.normal(scale=0.005, size=(len(q_surface_points), 3)),
        q_surface_points + np.random.normal(scale=0.007, size=(len(q_surface_points), 3)),
        q_surface_points + np.random.normal(scale=0.01, size=(len(q_surface_points), 3)),
    ]
    q_surface_points = np.concatenate(q_surface_points, axis=0)

    q_surface_points = np.concatenate((
        q_rand_points,
        q_surface_points, 
    ), axis=0)
    
    # SDF
    surface_sdf, _, _ = igl.signed_distance(q_surface_points, np.asarray(v), np.asarray(f), sign_type=igl.SIGNED_DISTANCE_TYPE_DEFAULT)

    np.savez("sdf.npz",
            q_surface_points=q_surface_points.astype(np.float32),
            surface_sdf=surface_sdf.astype(np.float32),
            )