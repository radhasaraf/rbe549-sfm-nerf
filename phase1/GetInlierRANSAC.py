import random

from EstimateFundamentalMatrix import *
from utils.data_utils import homogenize_coords,unhomogenize_coords

random.seed(42)


def get_inliers_RANSAC(v1,v2,max_iterations,threshold):
    """
    input:
        v1, v2 - non homogenous image feature coordinates
                 N x 2
        max_iterations - number of iterations it should run ransac for
        threshold - error threshold
    output:
        inliers - List[2, N x 2]
    """
    v1 = homogenize_coords(v1) # N x 3
    v2 = homogenize_coords(v2) # N x 3

    max_inliers = []
    for iter in range(max_iterations):
        # sample some random points assuming they are inliers
        sample_inds = random.sample(range(v1.shape[0]-1),8)
        sample_v1 = v1[sample_inds,:]
        sample_v2 = v2[sample_inds,:]

        # fit model
        _, F = estimate_fundamental_matrix(sample_v1, sample_v2) # 3 x 3

        error = F @ v1.T # 3 x N
        error = error.T # N x 3
        error = np.multiply(v2, error) # N x 3
        error = np.sum(error,axis=1) # N,

        # check if error is below some threshold for rest of the points
        curr_inliers = abs(error) < threshold # N,
        if np.sum(curr_inliers) > np.sum(max_inliers):
            max_inliers = curr_inliers

    v1_inliers = v1[max_inliers]
    v2_inliers = v2[max_inliers]

    # Un-homogenize
    v1_inliers = unhomogenize_coords(v1_inliers)
    v2_inliers = unhomogenize_coords(v2_inliers)

    return [v1_inliers, v2_inliers]
