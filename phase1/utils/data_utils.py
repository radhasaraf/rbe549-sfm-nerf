import numpy as np
def load_camera_instrinsics():
    """
    input:
        path - location from where calibration have to be loaded
    output:
        K - camera intrinsic matrix 3 x 3
            | alpha gamma u0 |
        K - |   0   beta  v0 |
            |   0     0    1 |
    """
    pass

# E = Tx X R (3 + 3) => 7
# R [r1,r2,r3] => r3 = r1 x r2 (3), ||r1|| = 1,||r2|| = 1 => 4
# scale invariant

def load_feature_matching_files():
    #TODO
    pass

def load_images(path, extn = ".png")
    """
    input:
        path - location from where files have to be loaded
        extn - file extension
    output:
        imgs - list of images - N
    """
    pass
