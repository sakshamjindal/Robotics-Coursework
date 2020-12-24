import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as sio
import cv2
import matplotlib.image as mpimg
from matplotlib import path
from scipy.spatial.transform import Rotation 
from mpl_toolkits.mplot3d import Axes3D

def vectorize_8_point(x1, x2):
    
    # Assume x1 = (u1,v1) and x2 =(u2,v2)
    u1 = x1[0]; u2 = x2[0]
    v1 = x2[1]; v2 = x2[1]

    return [
        u1*u2, u1*v2, u1, v1*u2, v1*v2, v1, u2, v2, 1
    ]

def enforce_singularity(F):
    
    U,S,Vd = np.linalg.svd(F, full_matrices=True)
    #Set the last elemet of the diagonal of S to  zero and re
    
    S[-1] = 0
    
    F = U@np.diag(S)@(Vd)

    return F

def eight_point_algorithm(matched_points1, matched_points2, M=None):
    '''
    Algorithm to compute Fundamental Matrix of Rank 2 via SVD for (x2)'F(x1) = 0 using exactly 8 correspondence points. The eight point
    algorithm is based on paper by <insert reference here>. Usually, we need an overdetermined set of equations
    
    Key Steps:
    1) Scale the data by diving each coordinate by M (maxium of the image's width and heing)
    2) Estimate Fundamental Matrix by SVD
    3) Enforce singularity condition of F
    4) Unscale the fundamental Matrix
    5) normalise the fundamental matrix
    
    :param matched_points1(np.array) : size(8x2) matching points in image1
    :param matched_points2(np.array) : size(8x2) matching points in image2
    :param M:scale factor to scale the matched points (taken here as max of width and height of the image)
    
    Returns: 
        Fundamental Matrix (np.array) : size(3x3) of rank2 such that
    '''
    
    if M is not None:
        matched_points1 = matched_points1/M
        matched_points2 = matched_points2/M
    
    A = []
    
    for x,y in zip(matched_points1,matched_points2):
        A.extend([vectorize_8_point(x,y)])
        
    A = np.array(A)

    U,S,Vd = np.linalg.svd(A, full_matrices=True)
        
    # extract from the last column of Vd.T
    F_ = Vd.T[:,-1].reshape(3,3)
    
    # there is a refine 'F' step here; Not sure about that. Need to study some theory here. 
    #print("Rank of the estimated Fundamental Matrix is : {}".format(np.linalg.matrix_rank(F_)))
        
    # F must be singular (remember, it is rank 2, since it isimportant for it to have a left and right nullspace, i.e. the epipoles).
    #print("Enforcing the Rank Constraint for the Fundamental Matrix ..")
    
    F = enforce_singularity(F_)
    #print("Rank of the re-estimated Fundamental Matrix is : {}".format(np.linalg.matrix_rank(F)))
    
    #unscale F by multiplying by the scale factor
    if M is not None:
        T = np.diag([1/M,1/M,1])
        F = T.T.dot(F).dot(T)
    
    #normalise the matrix such that ||F|| = 1
    F = F/np.linalg.norm(F)
    
    return F.reshape(3,3)    


def vectorize_point(point):
    """
    inputs a points (x,y) of shape (n) and converts into a vector form of shape (n,1) |x|
                                                                                      |y|
                                                                                      |1|
    """
    
    return np.hstack((point,np.array([0]))).reshape(-1,1)


def estimateFundamentalMatrix_RANSAC(matched_points1, matched_points2, num_iter=10000):
    """
    
    Implemented with the help of following implementation : https://github.com/AdityaNair111/RANSAC-based-scene-geometry/blob/dcc2a5df87b71bb8cf3bd26c3ae9cc66afec522f/code/code_geo.py#L157
    
    Find the best Fundamental Matrix using RANSAC on potentially matching points
    
    :param matched_points1(np.array) : size(Nx2) matching points in image1
    :param matched_points2(np.array) : size(Nx2) matching points in image2
    Each row is a correspondense matching points
    Returns:
        - best_F: (np.array) size (3,3) representing the best fundamental matrix
        - inlier_a: (np.array) size(M,2) representing the subset of corresponding points from image A that are inliers with respect to best_F
        - inliers_b: (np.array) size(M,2) representing the subset of corresponding points from image B that are inliers with respect to best_F
    """    
    
    assert matched_points1.shape==matched_points2.shape
    
    max_inliers = -np.inf
    num_iter = 10000
    point_algorithm = 8
    threshold = 0.001
    
    num_matches = matched_points1.shape[0]
    cost = np.zeros(num_matches)

    for i in range(num_iter):
        points_index=np.random.choice(num_matches, size=point_algorithm, replace=False)
        matches_a = matched_points1[points_index,:]; matches_b = matched_points2[points_index,:]
        F = eight_point_algorithm(matches_a, matches_b)
            
        inliers = np.zeros(num_matches, dtype=bool)
        num_inliers = 0
        
        for j in range(num_matches):
            cost=(vectorize_point(matched_points2[j]).T @ F @ vectorize_point(matched_points1[j])).item()
            
            if abs(cost)<threshold:
                num_inliers=num_inliers +1
                inliers[j]=True
        
        if num_inliers>max_inliers:
            max_inliers=num_inliers
            best_points_index=points_index
            best_inliers=inliers
            
    #max_inlers : number of points satisfying inlier criterion
    #best_points_index : set of 8 indexes used for computing the fundamentalmatrix
    #best_inliers : set of indexes satisfying the inlier criterion in the optimal condition
    
    inlier_pts1 = matched_points1[best_inliers,:]
    inlier_pts2 = matched_points2[best_inliers,:]
    
    best_F = eight_point_algorithm(inlier_pts1, inlier_pts2)
    
    return best_F, inlier_pts1, inlier_pts2