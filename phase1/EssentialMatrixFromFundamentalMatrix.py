import numpy as np
import cv2
from EstimateFundamentalMatrix import get_epipoles
from utils.visualization_utils import plot_features

def essential_from_fundamental(K, F, args):
    """
    input: 
        K - intrinsics - 3 x 3
        F - fundamental_matrix - 3 x 3
    output:
        E - essential_matrix - 3 x 3
    """
    #estimate KT@F@K
    E = K.T @ F @ K

    #take SVD of E
    UE, sigmaE, VE = np.linalg.svd(E)

    #modify and reestimate E from (1,1,0) condition
    corrected_sigmaE = np.array([1, 1, 0])
    reestimatedE = UE @ np.diag(corrected_sigmaE) @ VE
    if args.debug:
        print(f"before sigmaE:{sigmaE}")
        print(f"wo rank2 E:{E}")
        print(f"w rank2 E:{reestimatedE}")
        print(f"w rank2 E rank:{np.linalg.matrix_rank(reestimatedE)}")
    
    return reestimatedE

def test_E(K, F, E, img1, img2, window_name):
    """
    test whether epipoles from E and F are almost same
    """
    Fe1,Fe2 = get_epipoles(F, True)
    Ee1, Ee2 = get_epipoles(E, True)

    Fe1_r = K @ Ee1
    Fe2_r = K @ Ee2
    #TODO how are they exactly same till 1000th decimal point? 
    print(f"from F:{Fe1}")
    print(f"from E:{Fe1_r}")
    print(f"from F:{Fe2}")
    print(f"from E:{Fe2_r}")
    img1_copy = img1.copy()
    plot_features(img1_copy, [Fe1[0:2]], color=(255,0,0),thickness=20)
    plot_features(img1_copy, [Fe1_r[0:2]])

    img2_copy = img2.copy()
    plot_features(img2_copy, [Fe2[0:2]], color=(255,0,0),thickness=20)
    plot_features(img2_copy, [Fe2_r[0:2]])

    concat = np.hstack((img1_copy,img2_copy))
    cv2.imshow(f"{window_name}", concat)

