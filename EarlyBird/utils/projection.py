import numpy as np


def image_points_to_world(image_points, intrinsic, extrinsic):
    """Project Nx2 image points onto the z=0 ground plane."""
    image_points = np.asarray(image_points, dtype=np.float64)
    if image_points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    image_points = image_points.reshape(-1, 2)
    project_mat = np.asarray(intrinsic) @ np.asarray(extrinsic)
    ground_homography = np.delete(project_mat, 2, axis=1)
    image_h = np.concatenate([image_points.T, np.ones((1, len(image_points)))], axis=0)
    world_h = np.linalg.inv(ground_homography) @ image_h
    valid = np.abs(world_h[2]) > 1e-9
    world = np.full((len(image_points), 2), np.nan, dtype=np.float64)
    world[valid] = (world_h[:2, valid] / world_h[2:3, valid]).T
    return world
