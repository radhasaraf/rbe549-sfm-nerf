
def get_camera_extr_using_linear_pnp(v, world_points, K):
    """
    Returns new camera extrinsics (R and T) wrt 1st camera.
    inputs:
        v - N x 2 - features 
        X - N x 3 - world_points 
        K - 3 x 3 - camera matrix
    outputs:
        R: 3x3 
        T: 3x1
    """
    # Assuming image features and world points have a 1-1 correspondence 
    # Form equations with all features
    X, Y, Z = world_points.T
    x, y = v.T
    A1 = [X, Y, Z, ones, zeros, zeros, zeros, zeros, -x*X, -x*Y, -x*Z, -x] # N x 12
    A2 = [zeros, zeros, zeros, zeros, X, Y, Z, ones, -y*X, -y*Y, -y*Z, -y] # N x 12
    A  = np.hstack([A1, A2]) # 2*N x 12
    print(A.shape)
    exit(1)
    # Solve lin sys using SVD
    # Get R using K_inv * [p1, p2, p3]
    # Get gamma (1st singular value)
    # Enforce orthogonality on R: Singular values should be 1
    # Get T K_inv*p4/gamma
    pass
