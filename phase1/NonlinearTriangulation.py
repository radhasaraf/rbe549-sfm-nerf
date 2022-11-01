import numpy as np
from utils.helpers import homogenize_coords, unhomogenize_coords
from scipy.optimize import least_squares

def reprojection_error(v, X, P):
    """
    inputs:
        v - 3, - homogenized feature points
        X - 3, - world points
        P - 3 x 4 - projection matrix
    outputs:
        error - 2,
    """
    v_hat = P @ X.T # 3,
    v_hat = v_hat / v_hat[2] # 3,
    error = v_hat - v # 3,
    error = error[0:2]**2 # 2,
    #error = np.linalg.norm(error,axis=0,ord=2) # 1,
    return error

def compute_residuals(X, v1, v2, P1, P2, feature_point_index):
    """
    inputs:
        X  - 3, - world points to be optimized
        v1 - N x 3 - feature points 1st image
        v2 - N x 3 - feature points 2nd image
        P1 - 3 x 4 - projection matrix 1st image
        P2 - 3 x 4 - projection matrix 2nd image
        feature_point_index - index of the feature to be optimized
    outputs:
        error - 1,
    """
    error_1 = reprojection_error(v1[feature_point_index,:], X, P1) #2,
    error_2 = reprojection_error(v2[feature_point_index,:], X, P2) #2, 
    errors = np.concatenate((error_1,error_2)) # 2,
    return errors

def refine_triangulated_coords(K, C1, R1, C2, R2, v1, v2, X0):
    """
    inputs:
        K  - 3 x 3 - camera calibration matrix
        C1 - 3,    - 1st camera translation
        R1 - 3 x 3 - 1st camera rotation 
        C2 - 3,    - 2nd camera translation
        R2 - 3 x 3 - 2nd camera rotation
        v1 - N x 2 - feature points
        v2 - N x 2 - feature points
        X0 - N x 3 - initial world points estimated through linear triangulation
    outputs:
        world_coords
    """
    ## homogenize image and world coords
    v1 = homogenize_coords(v1) # N x 3
    v2 = homogenize_coords(v2) # N x 3
    X0 = homogenize_coords(X0) # N x 4

    # Compute P: cam x extr for both cams, making sure to consider T wrt to camera
    C1 = C1.reshape((3,1))
    T1 = - R1 @ C1 # 3 x 1
    P1 = K @ np.hstack((R1, T1)) # (3 x 3) @ (3 x 4) = 3 x 4

    C2 = C2.reshape((3,1))
    T2 = - R2 @ C2 # 3 x 1
    P2 = K @ np.hstack((R2, T2)) # (3 x 3) @ (3 x 4) = 3 x 4

    # Call scipy.optimize.least_squares which will take residual function
    X_optimized = []
    for i,X0_i in enumerate(X0):
        kwargs1 = {
                    "v1":v1,
                    "v2":v2,
                    "P1":P1,
                    "P2":P2,
                    "feature_point_index":i,
                }

        # Get refined coords from optimized param vector
        result = least_squares(compute_residuals, x0=X0[i,:], method='lm', kwargs=kwargs1)
        X_optimized.append(result.x)

    X_optimized = np.vstack(X_optimized)
    X_optimized = X_optimized/X_optimized[3]
    X_optimized = unhomogenize_coords(X_optimized)
    return X_optimized
