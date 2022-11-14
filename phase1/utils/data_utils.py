import csv
import glob
from typing import List, Tuple

import cv2
import numpy as np
import pprint
import random


def load_images(path, extn = ".png"):
    """
    input:
        path - location from where files have to be loaded
        extn - file extension
    output:
        images - list of images - N
    """
    img_files = glob.glob(f"{path}/*{extn}",recursive=False)

    img_names = []
    for img_file in img_files:
        img_name = img_file.rsplit(".", 1)[0][-1]
        img_names.append(img_name)

    imgs = {int(img_name) : cv2.imread(img_file) for img_file, img_name in zip(img_files,img_names)}
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
    K = np.array(K)
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

    D = {}
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
                    j = row[idx*3 + 6]
                    uj, vj = float(row[idx*3 + 7]), float(row[idx*3 + 8])

                    key = (int(i), int(j))
                    value = np.array([(ui,vi,uj,vj)])

                    D[key] = np.vstack((D[key],value)) if key in D else value

    newD = {}
    for key,value in D.items():
        v1,v2 = np.split(value, 2, axis=1)
        newD[key] = [v1,v2]

    return newD

class SFMMap():
    def __init__(self, path_to_matching_files: str) -> None:
        self.path = path_to_matching_files
        self.features_u, self.features_v = None, None
        self.visibility_matrix = None
        self._load()

        self.world_points = np.empty((self.visibility_matrix.shape[0],3))
        self.world_points.fill(np.nan)

    def _load(self) -> None:
        """
        inputs:
            path: path to matching files
        output:
            features u coords: N x num_of_images
            features v coords: N x num_of_images
            visibility matrix: N x num_of_images
        """
        feats_u, feats_v, visibility_mat = [], [], []

        # get a list of all matching*.txt files
        matching_files = glob.glob(f"{self.path}/matching*.txt", recursive=False)

        match_file_names = []
        for match_file in matching_files:
            file_name = match_file.rsplit(".", 1)[0][-1]
            match_file_names.append(file_name)

        matching_files = ['./Data/matching1.txt', './Data/matching2.txt', './Data/matching3.txt', './Data/matching4.txt']
        match_file_names = ['1','2','3','4']

        num_images = len(match_file_names) + 1
        for ith_cam, match_file in zip(match_file_names, matching_files):
            with open(match_file) as file:

                reader = csv.reader(file, delimiter=' ')
                for row_idx, row in enumerate(reader):

                    feat_u = np.zeros((1, num_images))
                    feat_v = np.zeros((1, num_images))
                    visibility = np.zeros((1, num_images), dtype=bool)

                    # ignoring the first line in each file
                    if row_idx == 0:
                        continue

                    # read first value and initialize the internal loop
                    n_matches_wrt_curr = int(row[0]) - 1

                    # convert camera number to array index
                    i = int(ith_cam) - 1

                    # read current feature coords
                    ui, vi = float(row[4]), float(row[5])

                    visibility[0, i] = True
                    feat_u[0, i] = ui
                    feat_v[0, i] = vi

                    # read j and subsequent feature coords in j
                    for idx in range(n_matches_wrt_curr):
                        jth_cam = row[3 * idx + 6]

                        # convert camera number to array index
                        j = int(jth_cam) - 1

                        uj = float(row[3 * idx + 7])
                        vj = float(row[3 * idx + 8])

                        # key = (int(i), int(j))
                        visibility[0, j] = True
                        feat_u[0, j] = uj
                        feat_v[0, j] = vj

                    feats_u.append(feat_u)
                    feats_v.append(feat_v)
                    visibility_mat.append(visibility)

        self.features_u = np.vstack(feats_u)
        self.features_v = np.vstack(feats_v)
        self.visibility_matrix = np.vstack(visibility_mat)

    def get_feat_matches(self, img_pair, num_of_samples= -1):
        """
        Returns all feature matches betw given image pair unless num_of_samples
        is provided in which case it randomly returns that many samples from the
        image features
        inputs:
            img_pair: (i, j)
            num_of_samples: int (If -1, return all)
        outputs:
            features in img i: num_of_samples x 2
            features in img j: num_of_samples x 2
            sample indices: num_of_samples,  # indices from visibility mat
        """

        ith_view, jth_view = img_pair
        i, j = ith_view - 1, jth_view - 1

        # Get features common in i and j
        idxs = np.where(
            np.logical_and(
                self.visibility_matrix[:, i],
                self.visibility_matrix[:, j]
            )
        )[0]

        # Get num_of_samples from common features
        if num_of_samples > 0:
            idxs = np.random.sample(idxs, num_of_samples)  # num_of_samples,

        vi = [self.features_u[idxs, i], self.features_v[idxs, i]]  # list(N, , N,)
        vj = [self.features_u[idxs, j], self.features_v[idxs, j]]  # list(N, , N,)

        vi = np.vstack(vi).T  # 2 x N -> N x 2
        vj = np.vstack(vj).T  # 2 X N -> N x 2

        return vi, vj, idxs


    def remove_matches(self, img_pair, outlier_idxs) -> None:
        """
        inputs:
            img_pair: (i, j)
        """
        _, jth_view = img_pair
        j = jth_view - 1

        self.visibility_matrix[outlier_idxs, j] = False

    def get_2d_to_3d_correspondences(self, ith_view):
        """
        inputs:
            ith_view: view for which we need 2D<->3D correspondences
        outputs:
            v: M x 2 - features
            X: M x 3 - corresponding world points
        """
        i = ith_view - 1

        # get indices from world_points where it is not nan
        indices_world = np.argwhere(~np.isnan(self.world_points[:,0])).flatten() # M1,

        # we will get indices for ith_view from visibility_matrix
        indices_visibility = np.where(self.visibility_matrix[:,i])[0] # M2,

        # find intersection of indices_world and indices_visibility
        indices = np.array(list(set(indices_world) & set(indices_visibility)))

        v = [self.features_u[indices, i], self.features_v[indices, i]]
        v = np.vstack(v).T  # M x 2

        X = self.world_points[indices]  # M x 3
        return v, X
    def add_world_points(self, world_points, indices):
        """
        inputs:
            world_points: M x 3
            indices: M,
        outputs:
            None
        """
        self.world_points[indices] = world_points

# One data structure that contains features, feat matches, world coords,
# visisibility matrix info.

# Usecases

# Load and get visibility mat
# Feat matches: Given i, j, num_of_samples -> feat_i, feat_j (-1: all)

# 2D-3D correspondences: Given i, num_of_samples -> img_i, world_i (-1: all)
# Updating the visibility matrix: Update True/False given image correspondences
# Update world coordinates post triangulation
# Getter(visibility) for BA


# 2d to 3d correspondences
# with sfm_map
# before that we need to add the world points into sfm map
# 3 pairs of images and the common correspondences

# we need 2D to 3D correspondences for 3
# we already have 2D to 3D correspondeces between 1 and 2
# we will get 2D to 3D for 3 using common features between 1, 2 and 3
