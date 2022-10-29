import numpy as np

def estimate_fundamental_matrix(v1, v2):
    """
    input:
        v1 - N x 3
        v2 - N x 3
    output:
        F - fundamental matrix 3 x 3
    """
    # construct Ax = 0
    x1, y1 = v1[:,0], v1[:,1] # N,
    x2, y2 = v2[:,0], v2[:,1] # N,
    ones = np.ones(x1.shape[0])

    A = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, ones] # N x 9
    A = np.vstack(A).T # N x 9

    # get SVD of A
    U,sigma,V = np.linalg.svd(A) # N x N, N x 9, 9 x 9
    f = V[np.argmin(sigma),:] # 9,

    # reconstruct F from singular vector
    F = np.array([
            [f[0],f[3],f[6]],
            [f[1],f[4],f[7]],
            [f[2],f[5],f[8]]
        ])

    # take SVD of F
    UF, sigmaF, VF = np.linalg.svd(F)
    sigmaF[2] = 0 # enforcing rank 2 constraint
    reestimatedF = UF @ np.diag(sigmaF) @ VF
    return reestimatedF
    # plot the epipolar lines with this F
    #   expected output: all epipolar do not intersect at one point (or no epipole)
    # set 3rd element of D to 0
    # reestimate F with new D
    # make sure that rank of reestimated F is 2
    # plot the epipolar lines with reestimated F
