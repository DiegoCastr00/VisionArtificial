import numpy as np 
def rgb2grayInverse(rgb):
        return np.dot(rgb[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
    
def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)