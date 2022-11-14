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
from LinearPnP import get_camera_extr_using_linear_pnp
from PnPRANSAC import refine_extr_using_PnPRANSAC
from BundleAdjustment import perform_BundleAdjustment
# from matplotlib import pyplot as plt
from ShowOutputs import *

from utils.helpers import homogenize_coords


def get_2d_to_3d_correspondences(D, i, Rs, Cs, K):
    """
    Returns the image and world points for the ith image based on matches and
    prior camera poses.

    inputs:
        D - data structure
        i - index on for which we need to get world correspondences, starts with 1
        Rs - List[i-1; 3 x 3] list of rotation matrices
        Cs - List[i-1; 3 x 1] list of camera poses
    outputs:
        v - N x 2 - features N x 2
        X - N x 3 - world points on image N x 3
    """
    image_points, world_points = [], []

    for j in range(1,i):
        vj, vi = D[(j,i)]  # N x 2, N x 2
        vj = homogenize_coords(vj).T  # 3 x N
        xj = np.linalg.inv(K) @ vj  # 3 x 3 @ 3 x N
        X = Rs[j-1].T @ xj + Cs[j-1].reshape((3, -1))  # 3 x N
        world_points.append(X.T)  # List(N x 3)
        image_points.append(vi)

    image_points = np.vstack(image_points)
    world_points = np.vstack(world_points)

    return image_points, world_points

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
    sfm_map = SFMMap(base_path)

    img_pairs = [
        (1, 2), (1, 3), (1, 4), (1, 5), (2, 3),
        (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)
    ]

    if args.debug:
        test_key = (1,2)
        matches_before = sfm_map.get_feat_matches(test_key)


    for pair in img_pairs:
        # Refine matches using RANSAC
        vi, vj, orig_idxs = sfm_map.get_feat_matches(pair)
        inlier_idxs = perform_RANSAC(vi, vj, 1000, 0.5)

        inlier_orig_indices = orig_idxs[np.where(inlier_idxs)[0]]
        outlier_set = set(orig_idxs) - set(inlier_orig_indices)
        outlier_idxs = np.array(list(outlier_set))  # Get outliers (idxs, inlier_idxs)

        sfm_map.remove_matches(pair, outlier_idxs)

    if args.debug:
        matches_after = sfm_map.get_feat_matches((1,2))
        show_before_after_RANSAC(imgs, test_key, matches_before, matches_after)
        # show_sample_matches_epipolars(imgs, test_key, matches_after)

    # estimate Fundamental matrix (F)
    F = get_ij_fundamental_matrix(1, 2, sfm_map)
    print(f"Fundamental_matrix_12:{F}")
    if args.debug:
        matches_after = sfm_map.get_feat_matches((1,2))
        show_epipolars(imgs[1], imgs[2], F, matches_after, f"epipolars_1_2")

    # estimate essential matrix E from Fundamental matrix F
    E = essential_from_fundamental(K, F, args)
    print(f"Essential_matrix_12:{E}")
    if args.debug:
        test_E(K, F, E, imgs[1], imgs[2], "epipoles_from_E_and_F")

    # extract camera pose from essential matrix
    Cs, Rs = extract_camera_pose(E)

    # triangulate the feature points to world points using camera poses
    v1, v2, orig_idxs_12 = sfm_map.get_feat_matches((1,2))
    Xs_all_poses = []

    C0 = np.zeros(3)
    R0 = np.eye(3)

    for C,R in zip(Cs,Rs):
        Xs = triangulate_points(K, C0, R0, C, R, v1, v2)
        Xs_all_poses.append(Xs)

    # disambiguate the poses using chierality condition
    C, R, X_linear, mod_idxs_12 = disambiguate_camera_poses(
        Cs, Rs, Xs_all_poses, orig_idxs_12
    )

    # remove the outliers from sfmMap
    outlier_idxs_12 = np.array(list(set(orig_idxs_12) - set(mod_idxs_12)))
    sfm_map.remove_matches((1,2), outlier_idxs_12)

    # perform non-linear triangulation
    X_non_linear = refine_triangulated_coords(K, C0, R0, C, R, v1, v2, X_linear)

    # register world points from 1st and 2nd camera view in the SFMMap
    sfm_map.add_world_points(X_non_linear, mod_idxs_12)

    if args.debug:
        show_disambiguated_and_corrected_poses(Xs_all_poses, X_linear, X_non_linear, C)

    r_mats = [np.eye(3), R]
    t_vecs = [np.zeros(3), C]
    for ith_view in range(3, len(imgs)+1):
        img_pts, world_pts = sfm_map.get_2d_to_3d_correspondences(ith_view)

        R_new, C_new, _, _ = refine_extr_using_PnPRANSAC(
            img_pts, world_pts, K, 10000, 1000
        )

        r_mats.append(R_new)
        t_vecs.append(C_new)

        X_new_linear = triangulate_points(K, C0, R0, C_new, R_new, v1, img_points)
        X_new_non_linear = refine_triangulated_coords(K, C0, R0, C_new, R_new, v1, img_points, X_non_linear)

        R_new, C_new, X_all = perform_BundleAdjustment(sfm_map, C_new, R_new, K, ith_view)

    if args.display:
        show_pnp_poses(t_vecs)

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
