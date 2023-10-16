import cv2
import numpy as np
def alpha2rgb(ruta):
    image = cv2.imread('imagenesPruebas/circle.png', cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    image = np.array(image)
    bg = np.array([255, 255, 255])
    alpha = (image[:, :, 3] / 255).reshape(image.shape[:2] + (1,))
    image = ((bg * (1 - alpha)) + (image[:, :, :3] * alpha)).astype(np.uint8)
    return image