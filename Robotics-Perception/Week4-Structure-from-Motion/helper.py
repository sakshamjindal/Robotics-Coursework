import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as sio
import cv2
import matplotlib.image as mpimg
from matplotlib import path
from scipy.spatial.transform import Rotation 


# Utils for visualization of epipolar lines
def drawlines(img1,img2,lines,pts1,pts2,color):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,_ = img1.shape
    if color is False:
        img1 = cv2.cvtColor(img1.astype(np.uint8),cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def draw_epipolar_lines_using_FundamentalMatrix(F, img_left, img_right, pts_left, pts_right, figSize=(60,30), color=True):
    """
    Source : https://github.com/nikhitmago/structure-from-motion/blob/master/sfm.py
    
    
    Draw the epipolar lines given the fundamental matrix, left right images and left right datapoints
    
    :param F(np.array) : size(3x3); fundamental matrx
    :param img_left
    :param img_right
    :param pts_left: Nx2
    :param pts_right:Nx2
    
    Returns:
        img with drawn epipolar lines
    
    """
    lines1 = cv2.computeCorrespondEpilines(pts_right.reshape(-1,1,2),2,F)
    lines1 = lines1.reshape(-1,3)
    img4,img5 = drawlines(img_left, img_right, lines1, pts_left, pts_right,color)

    lines2 = cv2.computeCorrespondEpilines(pts_left.reshape(-1,1,2),1,F)
    lines2 = lines2.reshape(-1,3)
    img6,img7 = drawlines(img_right, img_left, lines2, pts_right, pts_left, color)

    plt.figure(figsize=figSize); plt.set_cmap('gray');
    
    plt.subplot(121);plt.imshow(img4);  plt.axis('off')
    plt.subplot(122);plt.imshow(img6); plt.axis('off')