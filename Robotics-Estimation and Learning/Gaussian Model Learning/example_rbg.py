# This is an example code for collecting ball sample colors using roipoly

import numpy as np
import skimage.io as io
from skimage.color import rgb2gray
from roipoly import RoiPoly
from matplotlib import pyplot as pltc

imagePath = "data/train"
samples = []

for k in range(15):
    
    img = io.imread(imagePath + "/{0:03}.png".format(k+1))
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    img_gray = rgb2gray(img)
    plt.imshow(img)

    my_roi = RoiPoly(color='r')
    mask = my_roi.get_mask(img_gray)

    plt.imshow(mask)
    plt.show()

    R = R[mask]

    G = G[mask]
    B = B[mask]
    pdb.set_trace()

    sample = np.array([R,G,B])

    samples.append(sample)


samples = np.array(samples)