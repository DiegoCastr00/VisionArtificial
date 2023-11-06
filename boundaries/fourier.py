import numpy as np
import cv2
def find_fourier_descriptorsRotation(image):
    # Encuentra los contornos en la imagen
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Encuentra el contorno más largo
    max_contour = max(contours, key=len)

    # Calcula la transformada de Fourier de los contornos
    contour_complex = [complex(c[0][0], c[0][1]) for c in max_contour]
    fourier_transform = np.fft.fft(contour_complex)
    
    fourierCh = []
    for i in range(2, len(fourier_transform)):
        fourierCh = np.append(fourierCh, abs(fourier_transform[i]) / abs(fourier_transform[1]))

    return fourierCh


def equalTolerance(r, n):
    tolerancia = 1e-9
    for i in range(len(n)):
        if abs(n[i] - r[i]) > tolerancia:
            print(f"Los elementos en la posición {i} no son iguales. n[{i}] = {n[i]}, r[{i}] = {r[i]}")

def escPeque(img):
    new_height = 56
    new_width = 56

    # Escalar la matriz binaria utilizando interpolación
    scaled_matrix = np.zeros((new_height, new_width))

    for i in range(new_height):
            for j in range(new_width):
                    xi = int((i / (new_height - 1)) * (img.shape[0] - 1))
                    yi = int((j / (new_width - 1)) * (img.shape[1] - 1))
                    x1, x2 = xi, min(xi + 1, img.shape[0] - 1)
                    y1, y2 = yi, min(yi + 1, img.shape[1] - 1)  # Corregido 'y' a 'yi'
                    dx, dy = xi - x1, yi - y1
                    value = (1 - dx) * (1 - dy) * img[x1, y1] + dx * (1 - dy) * img[x2, y1] + \
                            (1 - dx) * dy * img[x1, y2] + dx * dy * img[x2, y2]
                        
                    scaled_matrix[i, j] = value

    # Redondear los valores de la matriz escalada a 0 o 1
    scaled_matrix = np.round(scaled_matrix)
    scaled_matrix = scaled_matrix.astype(np.uint8)
    return scaled_matrix

def escalar(img, C):
    # Supongamos que 'digit_pixels' es la matriz binaria original y 'C' es el factor de escala
    C =  4# Factor de escala
    scaled_image = np.multiply(img, C)

    # Asegúrate de que la imagen escalada esté en el rango adecuado (generalmente 0-255 para imágenes de 8 bits)
    scaled_image = np.clip(scaled_image, 0, 255)
    scaled_image = scaled_image.astype(np.uint8)
    return scaled_image
