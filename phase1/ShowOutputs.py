import numpy as np
import cv2
from utils.visualization_utils import *
from utils.data_utils import *
from EstimateFundamentalMatrix import *
from GetInlierRANSAC import *
from matplotlib import pyplot as plt

def show_sample_matches_epipolars(imgs, test_key, matches):

    img1 = imgs[test_key[0]]
    img2 = imgs[test_key[1]]
    v1, v2, _ = matches

    random.seed(42)
    inliers = random.sample(range(v1.shape[0]-1),8)
    v1_sample = v1[inliers, :]
    v2_sample = v2[inliers, :]

    show_matches2(img1, img2,[v1_sample, v2_sample], f"test_matches_{test_key}")

    F, reestimatedF = estimate_fundamental_matrix(v1_sample,v2_sample)

    # plot the epipolar lines without rank 2 F
    # expected output: all epipolar do not intersect at one point (or no epipole)
    show_epipolars(img1, img2, F, [v1_sample, v2_sample], f"test_wo_rank2_{test_key}")

    # plot the epipolar lines with reestimated F
    # expected output: all epipolar intersect at one point
    show_epipolars(img1, img2, reestimatedF, [v1_sample, v2_sample], f"test_w_rank2_{test_key}")

def show_before_after_RANSAC(imgs, test_key, matches_before, matches_after):

    img1 = imgs[test_key[0]]
    img2 = imgs[test_key[1]]
    v1, v2, _ = matches_before
    v1_corrected, v2_corrected, _ = matches_after

    show_matches2(img1, img2,[v1, v2], f"before_RANSAC_{test_key}")
    show_matches2(img1, img2,[v1_corrected, v2_corrected], f"after_RANSAC_{test_key}")
    
def show_disambiguated_and_corrected_poses(Xs_all_poses, X_linear, X_non_linear):
    plt.figure("camera disambiguation")
    colors = ['red','brown','greenyellow','teal']
    for color, X_c in zip(colors, Xs_all_poses):
        plt.scatter(X_c[:,0],X_c[:,2],color=color,marker='.')

    plt.figure("linear triangulation")
    plt.scatter(X_linear[:, 0], X_linear[:, 2], color='skyblue', marker='.')

    plt.figure("nonlinear triangulation")
    plt.scatter(X_non_linear[:, 0], X_non_linear[:, 2], color='red', marker='x')
