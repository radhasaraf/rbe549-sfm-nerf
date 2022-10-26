
def get_camera_extr_using_pnp():
    """
    Returns new camera extrinsics (R and T) wrt 1st camera.
    inputs:
        features:
        world_points:
    outputs:
        R: 3x3 
        T: 3x1
    """
    # Assuming image features and world points have a 1-1 correspondence 
    # Form equations with all features
    # Solve lin sys using SVD
    # Get R using K_inv * [p1, p2, p3]
    # Get gamma (1st singular value)
    # Enforce orthogonality on R: Singular values should be 1
    # Get T K_inv*p4/gamma
    pass
