
def disambiguate_camera_poses(poses, world_points):
    """
    Finds the correct camera pose obtd from ExtractCameraPose.py
    inputs:
        poses: ([3 x 1, 3 x 3] x 4)
        world_points: N x 3
    outputs:
        pose: [3 x 1, 3 x 3]
    """
    # For all poses
        # Get r3 from pose
        # Get C from
        # For all points
            # Check cheirality for camera 2
            # Append point to inliers count
    # Retain max count
    # Return pose corresponding to max count
    pass