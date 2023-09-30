import numpy as np
import thresholding
def thresholdingGlobal(D, threshold): # 0 or 255 since the threshold
    upThreshold = []
    downThreshold = [] 
    width_D, height_D = D.shape
    thresholdImg = np.copy(D)
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