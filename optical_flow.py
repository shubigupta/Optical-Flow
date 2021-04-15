import numpy as np
import cv2
from skimage import transform as tf
import matplotlib.pyplot as plt

from helpers import *


def getFeatures(img,bbox):
    """
    Description: Identify feature points within bounding box for each object
    Input:
        img: Grayscale input image, (H, W)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Output:
        features: Coordinates of all feature points in first frame, (F, N, 2)
    Instruction: Please feel free to use cv2.goodFeaturesToTrack() or cv.cornerHarris()
    """
    if len(bbox.shape) == 2:
        bbox = bbox.reshape(1,bbox.shape[0],bbox.shape[1])
    bbox = bbox.astype(int)

    features = np.zeros((bbox.shape[0],25,2))
    for i in range(bbox.shape[0]):
        bbox_img = img[bbox[i,0,1] : bbox[i,1,1]+1 , bbox[i,0,0] : bbox[i,1,0]+1]
        corners = cv2.goodFeaturesToTrack(bbox_img,25,0.01,5)
        #corners = np.int32(corners)

        if corners is None:
            features[i] = 0
            continue
        x = corners[:,0,0] + bbox[i,0,0]
        y = corners[:,0,1] + bbox[i,0,1]
        # features[i,:,0] = x
        # features[i,:,1] = y
        features[i,:corners.shape[0],0] = x
        features[i,:corners.shape[0],1] = y
    return features


# def estimateFeatureTranslation(feature, Ix, Iy, img1, img2):
def estimateFeatureTranslation(feature, img1, img2):
    """
    Description: Get corresponding point for one feature point
    Input:
        feature: Coordinate of feature point in first frame, (2,)
        Ix: Gradient along the x direction, (H,W)
        Iy: Gradient along the y direction, (H,W)
        img1: First image frame, (H,W,3)
        img2: Second image frame, (H,W,3)
    Output:
        new_feature: Coordinate of feature point in second frame, (2,)
    Instruction: Please feel free to use interp2() and getWinBound() from helpers
    """
    win_size = 15
    
    win_left, win_right, win_top, win_bottom = getWinBound(img1.shape, feature[0], feature[1], win_size)
    
    img1_win = img1[ win_top:win_bottom , win_left:win_right ]
    img2_win = img2[ win_top:win_bottom , win_left:win_right ]

    dx_sum = 0
    dy_sum = 0

    for i in range(20):
        Ix,Iy = findGradient(img2_win, 5, 1)
                
        It = img2_win - img1_win
        
        # A = np.hstack((Ix.reshape(-1, 1), Iy.reshape(-1, 1)))
        # b = -It.reshape(-1, 1)
        # res = np.linalg.solve(A.T @ A, A.T @ b)

        A = np.zeros((2,2))
        A[0,0] = (Ix.reshape(-1,1)**2).sum()
        A[1,1] = (Iy.reshape(-1,1)**2).sum()
        A[0,1] = (Ix.reshape(-1,1) * Iy.reshape(-1,1)).sum()
        A[1,0] = A[0,1].copy()
        
        b = np.zeros((2,1))
        b[0,0] = -(Ix.reshape(-1,1) * It.reshape(-1,1)).sum()
        b[1,0] = -(Iy.reshape(-1,1) * It.reshape(-1,1)).sum()
        
        res = np.linalg.solve(A,b)

        dx = res[0,0]
        dy = res[1,0]

        dx_sum += dx
        dy_sum += dy

        img2_win = get_new_img(img2, dx_sum, dy_sum, feature[0], feature[1], win_size)

        # img2_shift = get_new_img(img2, dx_sum, dy_sum)
        # img2_win = img2_shift[win_top:win_bottom+1, win_left:win_right+1]

    # new_feature = feature.copy()
    # new_feature[0] += dx_sum
    # new_feature[1] += dy_sum

    # return new_feature
    return np.array([feature[0] + dx_sum,feature[1] + dy_sum])


def estimateAllTranslation(features, img1, img2):
    """
    Description: Get corresponding points for all feature points
    Input:
        features: Coordinates of all feature points in first frame, (F, N, 2)
        img1: First image frame, (H,W,3)
        img2: Second image frame, (H,W,3)
    Output:
        new_features: Coordinates of all feature points in second frame, (F, N, 2)
    """
    if len(features.shape) == 2:
        features = features.reshape(1,features.shape[0],features.shape[1])

    new_features = np.zeros(features.shape)

    for i in range(features.shape[0]):
        for j in range(features.shape[1]):
            if features[i,j,0] == 0 and features[i,j,1] == 0:
                continue
            # corr_point = estimateFeatureTranslation(features[i,j,:], Ix, Iy, img1, img2)
            corr_point = estimateFeatureTranslation(features[i,j,:], img1, img2)

            new_features[i,j,:] = corr_point

    # new_features = None      
    return new_features


def applyGeometricTransformation(features, new_features, bbox, img2_sh):
    """
    Description: Transform bounding box corners onto new image frame
    Input:
        features: Coordinates of all feature points in first frame, (F, N, 2)
        new_features: Coordinates of all feature points in second frame, (F, N, 2)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Output:
        features: Coordinates of all feature points in secon frame after eliminating outliers, (F, N1, 2)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Instruction: Please feel free to use skimage.transform.estimate_transform()
    """
    for i in range(features.shape[0]):
        
        non_zero_entries    = ~np.logical_and( (features[i,:,0]==0) ,(features[i,:,1] == 0) )
        features_p          = features[i,non_zero_entries,:]
        new_features_p      = new_features[i,non_zero_entries,:]
        tform               = tf.estimate_transform('similarity',features_p,new_features_p)
        
        bbox[i] = tform(bbox[i])

    return new_features, bbox


