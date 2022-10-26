
def refine_triangulated_coords(v1, v2, tri_coords):
    """
    inputs:
        v1: N x 2
        v2: N x 2
        tri_coords: N x 3
        camera_mat: 3 x 3
        extrinsic_mat: 3 x 4
    outputs:
        world_coords

    """
    # Homogenize tri_coords
    # Compute P: cam x extr for both cams, making sure to consider T wrt to camera
    # Define residual function
    # Call scipy.optimize.least_squares which will take residual function
    # Get refined coords from optimized param vector
    # Return refined coords

    # show reprojected coords in both images

    pass

def residual_func():
    pass
