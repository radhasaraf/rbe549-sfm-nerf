import numpy as np

def get_inliers_RANSAC():
    """
    input:
        x1, x2 - non homogenous image feature coordinates
                 N x 2
    output:
        F - fundamental matrix 3 x 3
    """
    # sample some random points assuming they are inliers
    # fit model
    # check if error is below some threshold for rest of the points
    #   if it is then add them inlier set
    # if count of prev inlier set is less than count of curr inlier set
    #   then replace prev set with curr set
    # reestimate F for all the inliers
    # should be able to plot outliers and inlier matches
    pass
