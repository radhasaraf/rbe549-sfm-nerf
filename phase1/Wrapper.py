import numpy as np
import cv2
import argparse
from utils.visualization_utils import *
from utils.data_utils import *
from EstimateFundamentalMatrix import *
from GetInlierRANSAC import *

from TestOutputs import *
    

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
        corrected_pair_feat_matches[key] = get_inliers_RANSAC(value[0],value[1],1000,0.05)

    if args.display:
        show_before_after_RANSAC(imgs, img_pair_feat_matches, corrected_pair_feat_matches)
        show_sample_matches_epipolars(imgs, corrected_pair_feat_matches)

    # estimate Fundamental matrix (F)
    #F = get_12_fundamental_matrix(corrected_pair_feat_matches)
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
