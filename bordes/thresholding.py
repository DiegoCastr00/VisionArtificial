import numpy as np
def thresholding(D, threshold):
    width_D, height_D= D.shape
    thresholdImg = np.copy(D)
    for row in range(width_D):
        for column in range(height_D):
            if(D[row][column]>threshold):
                thresholdImg[row][column] = 0
            else:
                thresholdImg[row][column] = 1
    return thresholdImg
