
def get_triangulated_coords(v1, v2, camera_mat, extrinsic_mats):
    """
    Returns the world coordinates for the features in both images using
    triangulation
    inputs:
        v1: N x 2
        v2: N x 2
        camera_mat: 3 x 3
        extrinsic_mats: Tuple(3 x 4, 3 x 4)
    outputs:
        world_points: N x 3
    """

    # homogenize image coords
    # Construct perspective projection matrix(P) intr x extr
    # Construct sys of lin. equations
    # Solve sys using SVD
    # Un-homogenize coords

    # sanity check: All Zs should be positive

    pass