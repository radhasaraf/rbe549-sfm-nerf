
def get_visibility_matrix():
    """
    input:
        visibility_mat: No of poses x N
    output:
        visibility_mat: No of poses x N
    """

    # D1[img_id] <- [features](1,2,32,5,12,)
    # D2[{img_id1,img_id2}] <- [correspondences:(img_feature_id1,img_feature_id2)]
    # generate visibility matrix
