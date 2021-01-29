import numpy as np
from bresenham import bresenham
from numpy import matmul as mm
from scipy.stats import mode
import tqdm

def particle_localization(ranges, scanAngles, Map, param):
    
    N, M = ranges.shape[1], 1200
    myPose = np.zeros((3, N))
    myResolution, myOrigin = param['resol'], param['origin']
    myPose[:,0] = param['init_pose'].flatten()
    map_threshold_low = mode(Map, None)[0] - .3
    map_threshold_high = mode(Map, None)[0] + .3
    resample_threshold, radius = .85, .048
    sigma_m = .029 * np.array([[1], [1], [2]])
    direction = myPose[2, 0]
    P = np.tile(myPose[:, 0], (1, M))
    W = np.tile(1/M, (1, M))
    
    lidar_global = np.zeros((ranges.shape[0], 2))
    
    for j in tqdm.tqdm(range(1, N)):
        P = np.tile(myPose[:, j-1].reshape(-1, 1),(1, M))
        R = radius
        P += np.random.normal(0, 1,(3, M))*(mm(sigma_m, np.ones((1, M))))
        P[0, :M] += R*np.cos(P[2, :M])
        P[1, :M] += R*np.sin(P[2, :M])
        W = np.tile(1/M,(1, M))
        P_corr = np.zeros((1, M))
        for i in range(M):
            lidar_global[:,0] = np.array([(ranges[:, j] * np.cos(scanAngles + P[2,i]).flatten() +
                      P[0,i]) * myResolution + myOrigin[0]]).astype(int)
            lidar_global[:,1] = np.array([(-ranges[:,j] * np.sin(scanAngles + P[2,i]).flatten() +
                      P[1,i]) * myResolution + myOrigin[1]]).astype(int)
            lidar_global[lidar_global[:, 0] < 1, 0] = myOrigin[0]
            lidar_global[lidar_global[:, 0] < 1, 1] = myOrigin[1]
            lidar_global[lidar_global[:, 1] < 1, 0] = myOrigin[0]
            lidar_global[lidar_global[:, 1] < 1, 1] = myOrigin[1]
            lidar_global[lidar_global[:, 0] > Map.shape[1] - 1, 0] = myOrigin[0]
            lidar_global[lidar_global[:, 0] > Map.shape[1] - 1, 1] = myOrigin[1]
            lidar_global[lidar_global[:, 1] > Map.shape[0] - 1, 0] = myOrigin[0]
            lidar_global[lidar_global[:, 1] > Map.shape[0] - 1, 1] = myOrigin[1]
            lidar_global = lidar_global.astype(int)
            corr_values = Map[lidar_global[:, 1], lidar_global[:, 0]]
            P_corr[0,i] = -3*np.sum(corr_values <= map_threshold_low) + 10 * np.sum(corr_values >= map_threshold_high)
        P_corr -= np.min(P_corr)
        
        W = W[:M] * P_corr/np.sum(P_corr)
        W /= np.sum(W)
        
        ind = np.argmax(W)
        myPose[:,j] = P[:,ind]
    
    return myPose