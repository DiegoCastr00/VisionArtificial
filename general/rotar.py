import cv2
import matplotlib.pyplot as plt

def rotar(ruta_imagen, angulo_rotacion):
    alto, ancho = ruta_imagen.shape[:2]
    centro = (ancho // 2, alto // 2)
    matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo_rotacion, 1.0)
    imagen_rotada = cv2.warpAffine(ruta_imagen, matriz_rotacion, (ancho, alto))
    return imagen_rotada