import numpy as np

def generic_RANSAC():
    """
    description:
        a generic RANSAC taken from WikiPedia
    input:
        data  – A set of observations.
        model – A model to explain observed data points.
        n     – Minimum number of data points required to estimate model parameters.
        k     – Maximum number of iterations allowed in the algorithm.
        t     – Threshold value to determine data points that are fit well by model.
        d     – Number of close data points required to assert that a model fits well to data
    output:
        bestFit - model parameters which best fit the data (or null if no good model is found)
    """
    iterations = 0
    bestFit = None
    bestErr = None
    for i in range(k):
        maybeInliers = random.sample()
        maybeModel = model(maybeInliers)
    pass

def get_inliers_RANSAC():
    """
    input:
        x1, x2 - non homogenous image feature coordinates
                 N x 2
    output:
        F - fundamental matrix 3 x 3
    """
    #
    pass
