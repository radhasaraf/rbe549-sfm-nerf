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
    # Define residual function
    # Use quaternion in parameter vector
    # Call scipy.optimize.least_squares which will take residual function
    # Get refined pose from optimized param vector
    # Return refined pose

    # show reprojected coords in both images
