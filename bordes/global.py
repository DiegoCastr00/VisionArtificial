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
    
    return thresholdImg