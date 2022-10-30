import cv2
import numpy as np
from utils.data_utils import homogenize_coords

def show_features(img, features, window_name, color=(0, 255, 0)):
    """
    input:
        img - image on which features have to be plotted
        features - list of features N x 2
        color - color of the feature
        marker - shape of the feature
    """
    img_copy = img
    for feat in features:
        cv2.drawMarker(img_copy, [int(feat[0]), int(feat[1])], color)

    #cv2.imshow(window_name, img_copy)

def show_matches(img1, img2, matches, window_name, color = (0,255,0)):
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
    corners_2  = matches[1].copy()
    # Shift horizontally
    corners_2[:, 0] += img1_copy.shape[1]

    for (x1, y1), (x2, y2) in zip(corners_1, corners_2):
        cv2.line(concat, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

    cv2.imshow(window_name, concat)

def show_matches2(img1, img2, matches, window_name, color = (0,255,0)):
    """
    input:
        img1 - first image
        img2 - second image
        matches - List[2, N x 2]
    """
    dmatches = []
    v1 = []
    v2 = []
    for i in range(matches[0].shape[0]):
        dmatches.append(cv2.DMatch(i,i,0))
        keypoint1 = cv2.KeyPoint(matches[0][i][0], matches[0][i][1], 15)
        keypoint2 = cv2.KeyPoint(matches[1][i][0], matches[1][i][1], 15)
        v1.append(keypoint1)
        v2.append(keypoint2)

    ret = np.array([])
    drew_image = cv2.drawMatches(img1=img1, keypoints1=v1,
                                 img2=img2, keypoints2=v2,
                        matches1to2=dmatches,outImg = ret,matchesThickness=1)

    cv2.imshow(window_name, drew_image)

def format_coord_list(points):
    """
    input:
        points - N x 2
    output:
        fpoints - List[Tuple[2, 2,]]
    """
    fpoints = []
    for point in points:
        fpoints.append((int(point[0]),int(point[1])))
    return fpoints

def get_first_point(lines, img_shape):
    """
    input:
        lines - 3 x N
        img_shape - 2,
    output:
        point - N x 2
    """
    x = np.full(shape=(lines.shape[1]),fill_value=img_shape[1])
    y = -lines[2] - lines[0]*img_shape[1]
    y = y/lines[1]
    X = np.vstack((x,y)).T # N x 2
    X = format_coord_list(X)
    return X

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
    X = format_coord_list(X)
    return X

def show_epipolars(img1, img2, F, features, window_name, line_color=(0, 0, 0)):
    """
    input:
        img1 - image on which features have to be plotted
        img2 - image on which features have to be plotted
        F - fundamental matrix of epipolar pairs 3 x 3
        features - List[2, N x 2]
        line_color - epipolar line colors
        feature_marker - shape of the feature
        epipole_marker - shape of epipole marker (#TODO if it exists on image)
    """
    # multiply each fundamental matrix with other image feature correspondence
    v1 = homogenize_coords(features[0]) # N x 3
    v2 = homogenize_coords(features[1]) # N x 3
    lines1 = F.T @ v2.T # (3 x 3 @ 3 x N) = 3 x N
    lines2 = F @ v1.T # (3 x 3 @ 3 x N) = 3 x N

    first_points1 = get_first_point(lines1,img1.shape) # N x 2
    first_points2 = get_first_point(lines2,img2.shape) # N x 2

    second_points1 = get_second_point(lines1) # N x 2
    second_points2 = get_second_point(lines2) # N x 2

    img1_copy = img1.copy()
    for first_point, second_point in zip(first_points1, second_points1):
        cv2.line(img1_copy, first_point, second_point, line_color, 1)

    show_features(img1_copy, v1, f"{window_name}_1")

    img2_copy = img2.copy()
    for first_point, second_point in zip(first_points2, second_points2):
        cv2.line(img2_copy, first_point, second_point, line_color, 1)

    show_features(img2_copy, v2, f"{window_name}_2")

    cv2.imshow(f"{window_name}_1", img1_copy)
    cv2.imshow(f"{window_name}_2", img2_copy)

def show_reprojection():
    pass


def show_camera():
    pass

def show_rotation():
    """
    plot new axis based on new rotation
    """
    pass
