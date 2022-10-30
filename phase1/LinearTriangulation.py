import numpy as np
from utils.helpers import homogenize_coords, unhomogenize_coords, skew

def triangulate_points(K, C1, R1, C2, R2, v1, v2):
    """
    Returns the world coordinates for the features in both images using
    triangulation
    inputs:
        K - camera calibration matrix - 3 x 3
        C1 - 1st camera translation - 3,
        R1 - 1st camera rotation - 3 x 3
        C2 - 2nd camera translation  - 3,
        R2 - 2nd camera rotation - 3 x 3
        v1- N x 2
        v2- N x 2
    outputs:
        world_points: N x 3
    """
    ## homogenize image coords
    v1 = homogenize_coords(v1) # N x 3
    v2 = homogenize_coords(v2) # N x 3

    ## Construct perspective projection matrix(P) intr x extr
    # calculating translation of camera pose with respect to camera frame 
    C1 = C1.reshape((3,1))
    T1 = -R1.T @ C1 # 3 x 1
    P1 = K @ np.hstack((R1, T1)) # (3 x 3) @ (3 x 4) = 3 x 4

    C2 = C2.reshape((3,1))
    T2 = - R2.T @ C2 # 3 x 1
    P2 = K @ np.hstack((R2, T2)) # (3 x 3) @ (3 x 4) = 3 x 4

    # Construct sys of lin. equations
    Xs = []
    for point1, point2 in zip(v1,v2):
        A1 = skew(point1) @ P1 # 3 x 4
        A2 = skew(point2) @ P2 # 3 x 4
        A = np.vstack((A1,A2)) # 6 x 4

        # Solve sys using SVD
        U,sigma,V = np.linalg.svd(A) # 6 x n, n x 4, 4 x 4
        X = V[np.argmin(sigma),:] # 4,
        X = X/X[3]  # 4,

        # Un-homogenize coords
        X = X[0:3]  # 3,
        Xs.append(X)

    Xs = np.vstack(Xs) # N x 3
    return Xs

    # sanity check: All Zs should be positive

