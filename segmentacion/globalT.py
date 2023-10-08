import numpy as np
from .thresholding import thresholding

def thresholdingGlobal(D, threshold): # 0 or 255 since the threshold
    upThreshold = []
    downThreshold = [] 
    width_D, height_D = D.shape
    for row in range(width_D):
        for column in range(height_D):
            if(D[row][column]>threshold):
                upThreshold.append(D[row][column])
            else:
                downThreshold.append(D[row][column])
    np.array(upThreshold)
    np.array(downThreshold)
    meanup = np.mean(upThreshold)
    meandown = np.mean(downThreshold)
    newThreshold = (meandown + meanup)/2
    imagen = thresholding(D,newThreshold)
    return imagen


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


import cv2
imagen = cv2.imread('rain.jpg')
img = np.array(imagen)
imggray= rgb2gray(img)
Otsu = thresholdingGlobal(imggray, 50)

import matplotlib.pyplot as plt
plt.imshow(Otsu, cmap='gray')
