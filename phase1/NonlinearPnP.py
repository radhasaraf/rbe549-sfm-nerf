import numpy as np
from utils.helpers import homogenize_coords, unhomogenize_coords
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as scipyRot

def reprojection_error(v, X, P):
    """
    inputs:
        v - N x 3, - homogenized feature points
        X - N x 3, - world points
        P - 3 x 4 - projection matrix
    outputs:
        error - 2,
    """
    v_hat = P @ X.T # 3 x N
    v_hat = v_hat / v_hat[2] # 3 x N
    error = v_hat - v # 3 x N
    error = error[0:2]**2 # 2 x N
    #error = np.linalg.norm(error,axis=0,ord=2) # 1,
    return error

def compute_residuals(x, v, X):
    quat = x[0:4]
    R = scipyRot.from_quat(quat).as_matrix()  # 3 x 3
    t = x[4:7] # 3,

    P = np.hstack([R, t]) # 3 x 4
    error = reprojection_error(v, X, P)  # 2 x N
    error = np.mean(error, axis=1)  # N, 
    return error

def refine_poses(v, tri_coords, proj_mat):
    """
    inputs:
        v: N x 2
        tri_coords: N x 3
        proj_mat: 3 x 4
    outputs:
        world_coords
    """
    # Assuming v and tri_coords have 1-1 correspondence

    # Homogenize tri_coords
    v = homogenize_coords(v)
    
    # get quat and translation
    quat  = scipyRot.from_matrix(proj_mat[:3,:3]).as_quat() # 4, 
    T = proj_mat[:,3]  # 3,
    
    # make them as parameters
    v = np.array([quat, T]).flatten() # 7,
    
    # special arguments would be the world coordinates
    kwargs1 = {
                "v":v,
                "X":X,
            }
    # Define residual function

    # Use quaternion in parameter vector

    # Call scipy.optimize.least_squares which will take residual function
    # Get refined pose from optimized param vector
    # Return refined pose

    # show reprojected coords in both images