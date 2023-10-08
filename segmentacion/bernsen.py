import numpy as np
def Bernsen(imagen, umbralMin, windowSize):
    biny = np.zeros_like(imagen)
    ratioW = windowSize / 2
    for x in range(imagen.shape[0]):
        for y in range(imagen.shape[1]):
            xi = max(0, x- ratioW)
            xf = min(biny.shape[1]-1, x+ratioW)
            yi = max(0, y- ratioW)
            yf = min(biny.shape[0]-1, y+ratioW)
            
            minV = imagen[int(xi), int(yi)]
            maxV = imagen[int(xi), int(yi)]
            
            for i in range(int(xi), int(xf)):
                for j in range(int(yi), int(yf)):
                    value = imagen[i, j]
                    if value < minV:
                        minV = value
                    if value > maxV:
                        maxV = value
            
            umbral = (minV + maxV) / 2
            if umbral >= umbralMin:
                biny[x][y] = 1
            else:
                biny[x][y] = 0
    
    return biny