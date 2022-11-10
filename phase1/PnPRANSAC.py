from LinearPnP import get_camera_extr_using_linear_pnp
import random
import numpy as np
from utils.helpers import homogenize_coords, unhomogenize_coords


def refine_extr_using_RANSAC(features, world_points, max_iters, threshold, K):
    """
    Refines and returns extrinsics, inlier set of features and world points
    inputs:
        features: N x 2
        world_points: N x 3
        max_iters: number of iterations it should run ransac for
        threshold: error threshold
        K: camera matrix
    outputs:
        R: 3 x 3
        T: 3 x 1
        feature_inliers: N x 2
        world_point_inliers: N x 3
    """
    homogenized_world_points = homogenize_coords(world_points)  # N x 4
    u, v = features.T
    max_inliers = []
    for iter in range(max_iters):
        # sample 6 random points assuming they are inliers
        sample_inds = random.sample(range(features.shape[0]-1), 6)
        sample_features = features[sample_inds,:]  # 6 x 2
        sample_world_points = world_points[sample_inds,:]  # 6 x 3

        # fit model
        R, C = get_camera_extr_using_linear_pnp(
            sample_features, sample_world_points, K
        )

        # Make sure to consider t in P wrt camera
        T = -R * C
        P = K @ np.hstack((R,T))
        P1, P2, P3 = P  # 4, 4, 4,

        u_num = P1 @ homogenized_world_points.T  # N,
        v_num = P2 @ homogenized_world_points.T  # N,
        denom = P3 @ homogenized_world_points.T  # N,

        u_ = u_num / denom
        v_ = v_num / denom

        error = (u - u_)**2 + (v - v_)**2

        # check if error is below some threshold for rest of the points
        curr_inliers = abs(error) < threshold # N,
        if np.sum(curr_inliers) > np.sum(max_inliers):
            max_inliers = curr_inliers

    feature_inliers = features[max_inliers]
    world_point_inliers = world_points[max_inliers]

    # reestimate R and C for all the inliers
    R, C = get_camera_extr_using_linear_pnp(
        feature_inliers, world_point_inliers, K
    )

    return R, C, feature_inliers, world_point_inliers

