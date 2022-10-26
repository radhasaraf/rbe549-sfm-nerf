import csv
from typing import List
def load_camera_intrinsics(path: str) -> List[List]:
    """
    input:
        path - location of calibration.txt
    output:
        K - camera intrinsic matrix 3 x 3
            | alpha gamma u0 |
        K - |   0   beta  v0 |
            |   0     0    1 |
    """
    K = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            K.append([float(row[i]) for i in range(3)])
    return K

# E = Tx X R (3 + 3) => 7
# R [r1,r2,r3] => r3 = r1 x r2 (3), ||r1|| = 1,||r2|| = 1 => 4
# scale invariant

def load_feature_matching_files():
    #TODO
    pass

def load_images(path, extn = ".png"):
    """
    input:
        path - location from where files have to be loaded
        extn - file extension
    output:
        images - list of images - N
    """
    pass

if __name__ == '__main__':
    load_camera_intrinsics("./Data/calibration.txt")
