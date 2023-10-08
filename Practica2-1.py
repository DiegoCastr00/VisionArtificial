import numpy as np 
from segmentacion.globalT import thresholdingGlobal

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

import cv2
imagen = cv2.imread('rain.jpg')
img = np.array(imagen)
imggray= rgb2gray(img)
Otsu = thresholdingGlobal(imggray, 160)