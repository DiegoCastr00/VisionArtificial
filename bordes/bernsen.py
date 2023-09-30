import numpy as np
def Bernsen(imagen, umbralMin, windowSize):
    bin = np.zeros_like(imagen)
    ratioW = windowSize / 2
    for x in range(imagen.shape[0]):
        for y in range(imagen.shape[1]):
            xi = max(0, x- ratioW)
            xf = min(bin.shape[1]-1, x+ratioW)
            yi = max(0, y- ratioW)
            yf = min(bin.shape[0]-1, y+ratioW)
            
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
                bin[x][y] = 1
            else:
                bin[x][y] = 0
    
    return bin

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)