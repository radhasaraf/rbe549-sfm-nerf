import numpy as np
import cv2
import argparse
from utils.visualization_utils import *
from utils.data_utils import *
from EstimateFundamentalMatrix import *
from GetInlierRANSAC import *



def main(args):
    base_path = args.basePath
    input_extn = ".png"
    calibration_file = args.calibrationFile

    # get images
    imgs, img_names = load_images(base_path,input_extn)

    # get camera calibration
    K = load_camera_intrinsics(f"{base_path}{calibration_file}")

    # get matches
    img_pair_feat_matches = load_and_get_feature_matches(base_path)

    # show features
    # if args.display:
    #     for img in imgs:
    #         show_features(img, )

    # show feat matches
    if args.display:
        for key, value in img_pair_feat_matches.items():
            show_matches(imgs[key[0]-1], imgs[key[1]-1], value, f"Before matches {key}", (0, 0, 255))

    # get inliers RANSAC for all images
    for key, value in img_pair_feat_matches.items():
        # print(f"key{key}")

        # print(f"before: {img_pair_feat_matches[key][0].shape}")
        img_pair_feat_matches[key] = get_inliers_RANSAC(value[0],value[1],1000,1e-3)
        # print(f"after: {img_pair_feat_matches[key][0].shape}")

    if args.display:
        for key, value in img_pair_feat_matches.items():
            show_matches(imgs[key[0]-1], imgs[key[1]-1], value, f"After matches {key}", (0, 255, 0))

    v12 = img_pair_feat_matches[(1,2)]
    F0_1 = estimate_fundamental_matrix(v12[0], v12[1])
    show_epipolars(imgs[0], F0_1, img_pair_feat_matches[(1,2)], "0_1")
    # estimate Fundamental matrix (F)
    # estimate essential matrix E from Fundamental matrix F

    if args.display:
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
