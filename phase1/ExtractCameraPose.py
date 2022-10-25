import numpy as np


def extract_camera_pose():
    """
    Return 4 camera poses
    input:
        essential_mat - 3 x 3
    output:
        poses: List[3 x 1, 3 x 3]
    """
    # SVD of E
    # estimate C1, C2, C3, C4
    # define W
    # estimate R1, R2, R3, R4
    # check if det(R) > 0 
    #   if not then -C and -R
    pass
