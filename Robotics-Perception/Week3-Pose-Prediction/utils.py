# A is 2n x 9 matrix for solving the set of homography equations 
# It is solved by Singular Value Decomposition

import numpy as np
from numpy import linalg


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