from linearPnP import get_camera_extr_using_pnp

def refine_extr_using_ransac():
    """
    inputs:
        features:
        world_points:
    outputs:
        R:
        T:
    """
    # sample 6 random points assuming they are inliers
    # fit model
    # check if error is below some threshold for rest of the points
    #   if it is then add them inlier set
    # if count of prev inlier set is less than count of curr inlier set
    #   then replace prev set with curr set
    # reestimate R and C for all the inliers
    pass
