import numpy as np
from bresenham import bresenham
from numpy import matmul as mm
import tqdm

def occupancy_grid_mapping(ranges, scanAngles, pose, param):
    
    myResol = param['resol'] 
    myMap = np.zeros(param['size'])
    myorigin = param['origin']
    nsensors = len(scanAngles)
    N = pose.shape[1]
    lo_occ,lo_free = param['lo_occ'],param['lo_free']
    lo_max,lo_min = param['lo_max'],param['lo_min']
    
    for j in tqdm.tqdm(range(N)):
        xrobot, yrobot, theta = pose[0,j], pose[1,j], pose[2,j]
        ixrobot = int(np.ceil(myResol*xrobot) + myorigin[0])
        iyrobot = int(np.ceil(myResol*yrobot) + myorigin[1])
        
        for sensor in range(nsensors):
            d = ranges[sensor, j]
            alpha = scanAngles[sensor]
            xocc = d*np.cos(theta + alpha) + xrobot
            yocc =- d*np.sin(theta + alpha) + yrobot
            ixocc = int(np.ceil(myResol * xocc)) + myorigin[0]
            iyocc = int(np.ceil(myResol* yocc)) + myorigin[1]
            free = np.array(list(bresenham(ixrobot,iyrobot, ixocc[0],iyocc[0])))
            myMap[ixocc,iyocc] += lo_occ
            for i in range(len(free)):
                myMap[tuple(free[i])] -= lo_free
    
    myMap = np.clip(myMap,lo_min,lo_max)
    
    return myMap
