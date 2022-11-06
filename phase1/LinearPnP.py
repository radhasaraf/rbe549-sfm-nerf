import numpy as np

def get_camera_extr_using_linear_pnp(features, world_points, K):
    """
    Returns new camera extrinsics (R and T) wrt 1st camera.
    # Note: Check what T is wrt
    inputs:
        features - N x 2
        world_points - N x 3
        K - 3 x 3 - camera matrix
    outputs:
        R: 3x3  # wrt camera
        C: 3x1  # wrt world
    """
    # Assuming image features and world points have a 1-1 correspondence
    zeros = np.zeros((world_points.shape[0]))
    ones = np.ones ((world_points.shape[0]))

    # Form equations with all features
    X, Y, Z = world_points.T # N, N, N,
    u, v = features.T # N, N,
    A1 = np.vstack(
        [X, Y, Z, ones, zeros, zeros, zeros, zeros, -u*X, -u*Y, -u*Z, -u]
    ).T # N x 12
    A2 = np.vstack(
        [zeros, zeros, zeros, zeros, X, Y, Z, ones, -v*X, -v*Y, -v*Z, -v]
    ).T # N x 12
    A  = np.vstack([A1, A2]) # 2*N x 12

    # Solve lin sys using SVD
    UA, sigmaA, VA = np.linalg.svd(A)
    P = VA[np.argmin(sigmaA), :]  # 12,
    P = P.reshape((3,4)) # 3 x 4

    # Get R using K_inv * [p1, p2, p3]
    R = np.linalg.inv(K) @ P[0:3,0:3] # 3 x 3
    UR, sigmaR, VR = np.linalg.svd(R)

    # Get gamma (1st singular value)
    scale = sigmaR[0]

    # Get T K_inv*p4/gamma
    T = 1/scale * np.linalg.inv(K) @ P[:,3] # 3 x 1
    C = -R.T * T

    return R, C
