from linearPnP import get_camera_extr_using_pnp

def refine_extr_using_ransac():
    """
    inputs:
        features: N x 2
        world_points: N x 3
    outputs:
        R: 3 x 3
        T: 3 x 1
    """
    # sample 6 random points assuming they are inliers
    # Make sure to consider t in P wrt camera
    # fit model
    # check if error is below some threshold for rest of the points
    #   if it is then add them inlier set
    # if count of prev inlier set is less than count of curr inlier set
    #   then replace prev set with curr set
    # reestimate R and C for all the inliers
    pass
