import numpy as np
import cv2
from utils.visualization_utils import *
from utils.data_utils import *
from EstimateFundamentalMatrix import *
from GetInlierRANSAC import *

def show_sample_matches_epipolars(imgs, feat_matches_D):
    key = (1,2)

    img1 = imgs[key[0]]
    img2 = imgs[key[1]]
    v1, v2 = feat_matches_D[key]

    random.seed(42)
    inliers = random.sample(range(v1.shape[0]-1),8)
    v1_sample = v1[inliers, :]
    v2_sample = v2[inliers, :]

    show_matches2(img1, img2,[v1_sample, v2_sample], f"test_matches_{key}")

    F, reestimatedF = estimate_fundamental_matrix(v1_sample,v2_sample)

    # plot the epipolar lines without rank 2 F
    # expected output: all epipolar do not intersect at one point (or no epipole)
    show_epipolars(img1, img2, F, [v1_sample, v2_sample], f"test_wo_rank2_{key}")

    # plot the epipolar lines with reestimated F
    # expected output: all epipolar intersect at one point
    show_epipolars(img1, img2, reestimatedF, [v1_sample, v2_sample], f"test_w_rank2_{key}")

def show_before_after_RANSAC(imgs, feat_matches_D, corrected_feat_matches_D):
    key = (1,2)

    img1 = imgs[key[0]]
    img2 = imgs[key[1]]
    v1, v2 = feat_matches_D[key]
    v1_corrected, v2_corrected = corrected_feat_matches_D[key]

    show_matches2(img1, img2,[v1, v2], f"before_RANSAC_{key}")
    show_matches2(img1, img2,[v1_corrected, v2_corrected], f"after_RANSAC_{key}")
    
