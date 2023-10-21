import cv2
import matplotlib.pyplot as plt

def rotar(ruta_imagen, angulo_rotacion):
    imagen = cv2.imread(ruta_imagen)
    alto, ancho = imagen.shape[:2]
    centro = (ancho // 2, alto // 2)
    matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo_rotacion, 1.0)
    imagen_rotada = cv2.warpAffine(imagen, matriz_rotacion, (ancho, alto))
    return imagen_rotada