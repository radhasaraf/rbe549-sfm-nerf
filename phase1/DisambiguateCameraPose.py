import numpy as np

def disambiguate_camera_poses(Cs, Rs, Xs):
    """
    Finds the correct camera pose obtd from ExtractCameraPose.py
    inputs:
        Cs - List[4; 3 x 1, 3 x 3] all possible translations
        Rs - List[4; 3 x 1, 3 x 3] all possible rotations
        Xs - List[4; N x 3] all possible Xs corresponding to the poses
    outputs:
        pose: [3 x 1, 3 x 3]
    """
    # For all poses
    correctC = None
    correctR = None
    correctX = None
    max_inliers = []
    for C, R, X in zip(Cs, Rs, Xs):
        # Get r3 from pose
        r3 = R[:,2] # 3 x 1

        # For all points check cheirality for camera 2
        C = C.reshape((3,1)) # 3 x 1

        # Cheirality check for Camera 1 and 2
        cond1 = X[:,2].T  # 1 x N
        cond2 = r3.T @ (X.T - C)  # 1 x N

        inliers = np.logical_and(cond1 > 0, cond2 > 0)

        if np.sum(inliers) > np.sum(max_inliers):
            correctC = C
            correctR = R
            correctX = X
            max_inliers = inliers

    return correctC, correctR, correctX
