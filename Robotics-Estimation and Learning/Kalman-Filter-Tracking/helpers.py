import numpy as np
from numpy import matmul as mm
from scipy.linalg import cho_factor,cho_solve

def rdivide(A,B):
    c,low = cho_factor(B.T)
    C = cho_solve((c,low),A.T).T
    return C
def ldivide(A,B):
    c,low = cho_factor(A)
    C = cho_solve((c,low),B)
    return C

def kalmanFilter(t,x,y,state,param,previous_t):
    
    """
    
    """
    dt = t - previous_t
    C = np.array([[1,0,0,0],[0,1,0,0]])
    A = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
    Af = np.array([[1,0,.33,0],[0,1,0,.33],[0,0,1,0],[0,0,0,1]])
    Sm = np.diag([.02,.001,.09,.01])
    R = np.diag([.002,.002])
    
    if previous_t < 0 :
        state = np.array([x,y,0,0])
        param['P'] = .1*np.eye(4)
        predictx = x
        predicty = y
        return predictx,predicty,state,param
    
    P = param['P']
    P = mm(mm(A,P),A.T)+Sm
    
    K = rdivide(mm(P,C.T),R+mm(mm(C,P),C.T))
    
    xt = state.T
    z = np.array([[x],[y]])
    x_hat = mm(A,xt).reshape(-1,1) + mm(K,z-mm(mm(C,A),xt).reshape(-1,1))
    x_f = mm(Af,xt).reshape(-1,1) + mm(K,z-mm(mm(C,Af),xt).reshape(-1,1))
    state = x_hat.T
    predictx,predicty = x_f[0],x_f[1]
    P -= mm(mm(K,C),P)
    param['P'] = P
    return predictx,predicty,state,param