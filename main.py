import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte
import os
import copy

from optical_flow import *

def objectTracking(rawVideo):
    """
    Description: Generate and save tracking video
    Input:
        rawVideo: Raw video file name, String
    Instruction: Please feel free to use cv.selectROI() to manually select bounding box
    """
    cap = cv2.VideoCapture(rawVideo)
    imgs = []
    frame_cnt = 0 

    # Initialize video writer for tracking video
    trackVideo = 'results/Output_' + rawVideo
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(trackVideo, fourcc, fps, size)
    
    # Define how many objects to track
    F = 3
                  
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret: continue
        # vis = frame.copy()
        vis = copy.deepcopy(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        frame_cnt += 1
        
        if frame_cnt == 1:
            bbox = np.zeros((F,2,2))
            
            # Manually select objects on the first frame
            for f in range(F):
                x,y,w,h = np.int32(cv2.selectROI("roi", vis, fromCenter=False))
                cv2.destroyAllWindows()
                bbox[f] = np.array([(x,y), (x+w, y+h)])
                
            features = getFeatures(frame, bbox)
            frame_old = frame.copy()

        else:

            new_features = estimateAllTranslation(features, frame_old, frame)

            for i in range(features.shape[0]):
                distance = np.sqrt((new_features[i,:,0] - features[i,:,0])**2 + (new_features[i,:,1] - features[i,:,1])**2)
                points_discard1 = (distance > 5)# + (new_features[i,:,0] > frame.shape[1]) + (new_features[i,:,1] > frame.shape[0])
                points_discard2 = (features[i,:,0] > bbox[i,1,0]) + \
                                  (features[i,:,0] < bbox[i,0,0]) + \
                                  (features[i,:,1] > bbox[i,1,1]) + \
                                  (features[i,:,1] < bbox[i,0,1])
                points_discard = points_discard1 + points_discard2
                new_features[i][points_discard,:] = 0
                features[i][points_discard,:] = 0

                zero_entries = np.logical_and( (features[i,:,0]==0) ,(features[i,:,1] == 0) ).sum()
                if zero_entries > 20:
                    # save_fig(F, bbox, vis, features, frame_cnt)
                    features[i] = getFeatures(frame,bbox[i])[0]
                    zero_entries = np.logical_and( (features[i,:,0]==0) ,(features[i,:,1] == 0) ).sum()
                    if zero_entries > 15:
                        features[i] = 0
                        bbox[i] = 0
                        new_features[i] = 0
                    # save_fig(F, bbox, vis, features, frame_cnt + 100)
                    new_features[i] = estimateAllTranslation(features[i], frame_old, frame)
            
            features, bbox = applyGeometricTransformation(features, new_features, bbox, frame.shape)
            
            for i in range(bbox.shape[0]):
                if bbox[i,1,0] > frame.shape[1]:
                    bbox[i,1,0] = frame.shape[1]
                if bbox[i,1,1] > frame.shape[0]:
                    bbox[i,1,1] = frame.shape[0]

            frame_old = frame.copy()
            
        # display the bbox
        for f in range(F):
            cv2.rectangle(vis, tuple(bbox[f,0].astype(np.int32)), tuple(bbox[f,1].astype(np.int32)), (0,0,255), thickness=2)
        
        # display feature points
        for f in range(F):
            for feature in features[f]:
                cv2.circle(vis, tuple(feature.astype(np.int32)), 2, (0,255,0), thickness=-1)
        
        # save to list
        imgs.append(img_as_ubyte(vis))
        
        # save image
        if (frame_cnt + 1) % 10 == 0:
            cv2.imwrite('results/{}.jpg'.format(frame_cnt), img_as_ubyte(vis))

        # Save video with bbox and all feature points
        writer.write(vis)
        
        # Press 'q' on the keyboard to exit
        cv2.imshow('Track Video', vis)

        if cv2.waitKey(30) & 0xff == ord('q'): break
        
        
    # Release video reader and video writer
    cv2.destroyAllWindows()
    cap.release()
    writer.release()
    
    return


if __name__ == "__main__":
    rawVideo = "Easy.mp4"
    if not os.path.exists("results"): os.mkdir("results")
    objectTracking(rawVideo)