import numpy as np
import cv2
import argparse
from utils.visualization_utils import *
from utils.data_utils import *
from EstimateFundamentalMatrix import *
from GetInlierRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import refine_triangulated_coords

from ShowOutputs import *

def main(args):
    base_path = args.basePath
    input_extn = ".png"
    calibration_file = args.calibrationFile
    random.seed(42)

    # get images
    imgs, img_names = load_images(base_path,input_extn)

    # get camera calibration
    K = load_camera_intrinsics(f"{base_path}{calibration_file}")

    # get matches
    img_pair_feat_matches = load_and_get_feature_matches(base_path)

    # get inliers RANSAC for all images
    corrected_pair_feat_matches = {}
    for key, value in img_pair_feat_matches.items():
        corrected_pair_feat_matches[key] = get_inliers_RANSAC(value[0],value[1],1000,0.090)

    if args.debug:
        show_before_after_RANSAC(imgs, img_pair_feat_matches, corrected_pair_feat_matches)
        show_sample_matches_epipolars(imgs, corrected_pair_feat_matches)

    # estimate Fundamental matrix (F)
    F = get_ij_fundamental_matrix(1, 2, corrected_pair_feat_matches)
    if args.display:
        show_epipolars(imgs[1], imgs[2], F, corrected_pair_feat_matches[(1,2)], f"epipolars_1_2")

    # estimate essential matrix E from Fundamental matrix F
    E = essential_from_fundamental(K, F, args)

    if args.debug:
        test_E(K, F, E, imgs[1], imgs[2], "epipoles_from_E_and_F")

    # extract camera pose from essential matrix
    Cs, Rs = extract_camera_pose(E)

    # triangulate the feature points to world points using camera poses
    v1, v2 = corrected_pair_feat_matches[(1,2)]
    Xs_all_poses = []

    for C,R in zip(Cs,Rs):
        Xs = triangulate_points(K, np.zeros(3), np.eye(3), C, R, v1, v2)
        Xs_all_poses.append(Xs)

    # disambiguate the poses using chierality condition
    C, R, X_linear = disambiguate_camera_poses(Cs, Rs, Xs_all_poses)

    # perform non linear triangulation
    X_non_linear = refine_triangulated_coords(K, np.zeros(3), np.eye(3), C, R, v1, v2, X_linear)

    if args.display:
        show_disambiguated_and_corrected_poses(Xs_all_poses, X_linear, X_non_linear)

    ## camera registration
    # perform LinearPnP
    # optimize using NonLinearPnP
    # new 3D points using Linear Triangulation
    # optimize new 3D points using NonLinear Triangulation
    # add new 3D points to whole data
    # Build Visibility matrix
    # perform Bundle Adjustment
    
    if args.debug or args.display:
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basePath',default='./Data/')
    parser.add_argument('--calibrationFile',default='calibration.txt')
    parser.add_argument('--display',action='store_true',help="to display images")
    parser.add_argument('--debug',action='store_true',help="to display debug information")

    args = parser.parse_args()
    main(args)
