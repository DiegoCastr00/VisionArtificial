import numpy as np 
from segmentacion.globalT import thresholdingGlobal
from segmentacion.otsu import thresholdOtsu

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

import cv2
imagen = cv2.imread('ejemplo.jpeg')
img = np.array(imagen)
Otsu = thresholdOtsu(rgb2gray(img))

import matplotlib.pyplot as plt
plt.imshow(Otsu, cmap='gray')