import cv2
import numpy as np
from utils.data_utils import homogenize_coords

def show_features(img, features, window_name, color=(255, 0, 0)):
    """
    input:
        img - image on which features have to be plotted
        features - list of features N x 2
        color - color of the feature
        marker - shape of the feature
    """
    img_copy = np.copy(img)
    for feat in features:
        cv2.drawMarker(img_copy, [feat[1], feat[0]], color)

    cv2.imshow(window_name, img_copy)

def show_matches(img1, img2, matches, window_name, color):
    """
    input:
        img1 - first image
        img2 - second image
        matches - List[2, N x 2]
    """
    img1_copy = img1.copy()
    img2_copy = img2.copy()

    concat = np.concatenate((img1_copy, img2_copy), axis=1)

    corners_1 = matches[0].copy()  # N x 2
    print(f"corners_1: {corners_1.shape}")
    corners_2  = matches[1].copy()
    # Shift horizontally
    corners_2[:, 0] += img1_copy.shape[1]

    for (x1, y1), (x2, y2) in zip(corners_1, corners_2):
        cv2.line(concat, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

    cv2.imshow(window_name, concat)

def get_second_point(lines):
    """
    input:
        lines - 3 x N
    output:
        point - N x 2
    """
    y = -lines[2] / lines[1] # N,
    x = np.zeros((lines.shape[1]))
    X = np.vstack((x,y)).T # N x 2
    return X

def show_epipolars(img, F, features, window_name, line_color=(0, 0, 0)):
    """
    input:
        img - image on which features have to be plotted
        F - fundamental matrix of epipolar pairs 3 x 3
        features - List[2, N x 2]
        line_color - epipolar line colors
        feature_marker - shape of the feature
        epipole_marker - shape of epipole marker (#TODO if it exists on image)
    """
    # multiply each fundamental matrix with other image feature correspondence
    v1 = homogenize_coords(features[0]) # N x 3
    v2 = homogenize_coords(features[1]) # N x 3
    lines1 = F @ v2.T # (3 x 3 @ 3 x N) = 3 x N
    lines2 = F @ v1.T # (3 x 3 @ 3 x N) = 3 x N

    second_points1 = get_second_point(lines1) # N x 2
    second_points2 = get_second_point(lines2) # N x 2

    img_copy = img.copy()
    print(f"lines1:{lines1.shape}")
    print(f"lines2:{lines2.shape}")
    print(f"second_points1:{second_points1.shape}")
    print(f"second_points2:{second_points2.shape}")
    for point,second_point in zip(v1,second_points1):
        cv2.line(img_copy, (int(point[0]),int(point[1])),(int(second_point[0]),int(second_point[1])), line_color, 1)

    cv2.imshow(window_name, img_copy)

def show_reprojection():
    pass


def show_camera():
    pass

def show_rotation():
    """
    plot new axis based on new rotation
    """
    pass
