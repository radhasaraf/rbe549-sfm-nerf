import numpy as np
def homogenize_coords(coords):
    """
    N x 2 -> N x 3
    """
    ret = np.concatenate((coords,np.ones((coords.shape[0],1))),axis=1)
    return ret

def unhomogenize_coords(coords):
    """
    Nx3 -> Nx2
    """
    ret = np.delete(coords, 2, axis=1)
    return ret
    
def skew(x):
    """
    3, -> 3 x 3
    """
    return np.array([[ 0  , -x[2],  x[1]],
                     [x[2],    0 , -x[0]],
                     [-x[1], x[0],    0]])

    

