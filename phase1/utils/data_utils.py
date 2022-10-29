import csv
import glob
from collections import defaultdict
from typing import List

import cv2
import numpy as np


def homogenize_coords(coords):
    """
    Nx2 -> Nx3
    """
    ret = np.concatenate((coords,np.ones((coords.shape[0],1))),axis=1)
    return ret

def load_images(path, extn = ".png"):
    """
    input:
        path - location from where files have to be loaded
        extn - file extension
    output:
        images - list of images - N
    """
    img_files = glob.glob(f"{path}/*{extn}",recursive=False)
    img_names = [img_file.replace(f"{path}/",'').replace(f"{extn}",'') for img_file in img_files]
    imgs = [cv2.imread(img_file) for img_file in img_files]
    return imgs, img_names

def load_camera_intrinsics(path: str) -> List[List]:
    """
    input:
        path - location of calibration.txt
    output:
        camera intrinsic matrix 3 x 3
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

def load_and_get_feature_matches(path):
    """
    Gives out a datastructure containing all img-img
        correspondences
    inputs:
        path - path from where it has to read
    outputs:
        img_to_img_matches - img correspondences
    """

    # get a list of all matching*.txt files
    matching_files = glob.glob(f"{path}/matching*.txt",recursive=False)

    match_file_names = []
    for match_file in matching_files:
        # file_name = match_file.rsplit(".", 1)[0][-1]
        file_name = match_file.rsplit(".", 1) # splits path into list of non-extension, extension parts
        file_name = file_name[0] # Gets the non-extension part
        file_name = file_name[-1] # Gets number of matching file
        match_file_names.append(file_name)

    D = defaultdict()

    for i, match_file in zip(match_file_names, matching_files):
        with open(match_file) as file:
            reader = csv.reader(file, delimiter=' ')

            for row_idx,row in enumerate(reader):
                # ignoring the first line in each file
                if row_idx == 0:
                    continue

                # read first value and initialize the internal loop
                n_matches_wrt_curr = int(row[0]) - 1

                # read current feature coords
                ui, vi = float(row[4]), float(row[5])

                # read j and subsequent feature coords in j
                for idx in range(n_matches_wrt_curr):
                    j = int(row[idx*3 + 6])
                    uj, vj = float(row[idx*3 + 7]), float(row[idx*3 + 8])

                    key = (i,j)
                    value = np.array([(ui,vi,uj,vj)])
                    if key in D:
                        D[key] = np.vstack((D[key],value))
                    else:
                        D[key] = value

    newD = defaultdict()
    for key,value in D.items():
        v1,v2 = np.split(value, 2, axis=1)
        newD[key] = [v1,v2]

    return newD
