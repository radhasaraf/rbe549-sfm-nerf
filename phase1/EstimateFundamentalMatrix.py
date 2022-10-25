
def estimate_fundamental_matrix(x1, x2):
    """
    input: 
        x1, x2 - non homogenous image feature coordinates
                 N x 2
    output:
        F - fundamental matrix 3 x 3
    """
    # construct Ax = 0
    # get SVD of A
    # get right most singular value # lookup homework1
    # take SVD of F
    # plot the epipolar lines with this F
    # set 3rd element of D to 0
    # reestimate F with new D
    # make sure that rank of reestimated F is 2
    # plot the epipolar lines with reestimated F
    pass
