import numpy as np
def get_R_from_matrix(M):
    UR, sigmaR, VR = np.linalg.svd(M)
    sigmaR = [1,1,1]
    R = UR @ np.diag(sigmaR) @ VR
    return R

def extract_camera_pose(E):
    """
    Return 4 camera poses
    input:
        E - 3 x 3
    output:
        C, R - List[4 ; 3 x 1, 3 x 3]
    """
    # SVD of E
    UE, sigmaE, VE = np.linalg.svd(E)

    # estimate C1, C2, C3, C4
    # possible translations from left null space of E
    C =  UE[:,2] 
    Cs = [C, -C, C, C]

    # estimate R1, R2, R3, R4
    # creating S (slide 7 lecture 14)
    S = np.array([
            [ 0,-1, 0 ],
            [ 1, 0, 0 ],
            [ 0, 0, 0 ]
        ])

    # possible rotations based on the above S
    R1 = UE @ S   @ VE
    R1 = get_R_from_matrix(R1)
    R2 = UE @ S.T   @ VE
    R2 = get_R_from_matrix(R2)
    Rs = [R1,R1,R2,R2]

    Cs_final = []
    Rs_final = []
    eps = 1e-2
    for C,R in zip(Cs,Rs):
        det_value = np.linalg.det(R)
        if -1 - eps < det_value and det_value < -1 + eps :
            Cs_final.append(-C)
            Rs_final.append(-R)
        else:
            Cs_final.append(C)
            Rs_final.append(R)

    return Cs_final, Rs_final
