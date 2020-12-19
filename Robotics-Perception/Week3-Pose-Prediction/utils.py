# A is 2n x 9 matrix for solving the set of homography equations 
# It is solved by Singular Value Decomposition

import numpy as np
from numpy import linalg
import cv2
from matplotlib import pyplot as plt
import math
import sys

num_frames = 166

def vectorize(p,q):
    x1=p[0]
    x2=p[1]
    x_1 = q[0]
    x_2 = q[1]
   
    return [[-x1, -x2, -1, 0, 0, 0, x1*x_1, x2*x_1, x_1],
            [0, 0 , 0, -x1, -x2, -1, x1*x_2, x2*x_2, x_2]]


def solve_homography(points_p, points_q):
    """
    Solves the homography equation using the set of equation that maps a set of points
    p = (p1,p2,1) to another set of points q = (q1, q2, 1) such that q ~ Hp
    """
    
    assert len(points_p)==4
    assert len(points_q)==4
    
    A = []
    
    for p,q in zip(points_p,points_q):
        A.extend(vectorize(p,q))
        
    A = np.array(A)

    # solving a set of equations such that Ah=0 and h = flatten(H)
    U,s,V = linalg.svd(A, full_matrices=True)
    
    #The vector h will then be the last column of V.T
    H = V[-1,:].reshape(3,3)
    
    return H

class KLTtrack():
    def __init__(self,imglist,trackpts):
        self.vid = imglist
        self.pts = trackpts
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize  = (31,31),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 5, 0.03))

    def pointTracker(self):
        # Create some random colors
        color = np.random.randint(0,255,(self.pts.shape[0],3))
        # Take first frame and find corners in it
        old_frame = self.vid[0,:,:,:]
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = np.floor(self.pts).reshape(-1,1,2).astype(np.float32)
        
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        corners = []
        corners.append(self.pts)

        for i in range(1,len(self.vid)):
            frame = video_imgs[i,:,:,:]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new, good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            img = cv2.add(frame,mask)

            k = cv2.waitKey(5) & 0xff
            if k == 27:
                break
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            corners.append(good_new)
            p0 = good_new.reshape(-1,1,2)
            
        return np.array(corners)
