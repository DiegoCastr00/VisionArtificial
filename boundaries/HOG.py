import math
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cv2

def hog(img):
    img = np.array(img, dtype=np.int64) 

    mag = []
    theta = []
    for i in range(28):
        magnitudeArray = []
        angleArray = []
        for j in range(28):
            # Condition for axis 0
            if j-1 <= 0 or j+1 >= 28:
                if j-1 <= 0:
                    # Condition if first element
                    Gx = img[i][j+1] - 0
                elif j + 1 >= len(img[0]):
                    Gx = 0 - img[i][j-1]
            # Condition for first element
            else:
                Gx = img[i][j+1] - img[i][j-1]

            # Condition for axis 1
            if i-1 <= 0 or i+1 >= 28:
                if i-1 <= 0:
                    Gy = 0 - img[i+1][j]
                elif i + 1 >= 28:
                    Gy = img[i-1][j] - 0
            else:
                Gy = img[i-1][j] - img[i+1][j]

            # Calcula la magnitud
            magnitude = math.sqrt(pow(Gx, 2) + pow(Gy, 2))
            # Asegura que la magnitud esté dentro del rango válido de int64
            magnitude = min(magnitude, np.iinfo(np.int64).max)
            magnitudeArray.append(round(magnitude, 9))

            # Calculating angle
            if Gx == 0:
                angle = math.degrees(0.0)
            else:
                angle = abs(math.degrees(math.atan(Gy / Gx)))
            angle = min(angle, np.iinfo(np.int64).max)
            angleArray.append(round(angle, 9))
        mag.append(magnitudeArray)
        theta.append(angleArray)

    mag = np.array(mag)  
    theta = np.array(theta)

    # plt.figure(figsize=(15, 8))
    # plt.imshow(mag, cmap="gray")
    # plt.axis("off")
    # plt.show()


    # plt.figure(figsize=(15, 8))
    # plt.imshow(theta, cmap="gray")
    # plt.axis("off")
    # plt.show()

    # Tamaño de la celda en píxeles (por lo general, 8x8)
    cell_size = 8

    # Número de celdas en el eje X e Y
    cells_x = len(mag[0]) // cell_size
    cells_y = len(mag) // cell_size

    # Número de orientaciones en el histograma (por lo general, 9)
    num_bins = 9

    # Inicializar histogramas
    histograms = np.zeros((cells_y, cells_x, num_bins), dtype=np.float32)

    # Rango de ángulos de orientación (de 0 a 180 grados)
    angle_range = 180

    # Calcular el ancho de cada bin del histograma
    bin_width = angle_range / num_bins

    # Para cada celda
    for y in range(cells_y):
        for x in range(cells_x):
            # Obtener las magnitudes y orientaciones en la celda
            cell_magnitudes = mag[y * cell_size:(y + 1) * cell_size, x * cell_size:(x + 1) * cell_size]
            cell_angles = theta[y * cell_size:(y + 1) * cell_size, x * cell_size:(x + 1) * cell_size]

            # Calcular histograma de orientación de gradientes para la celda
            for i in range(cell_size):
                for j in range(cell_size):
                    angle = cell_angles[i, j]
                    magnitude = cell_magnitudes[i, j]

                    # Calcular a qué bin pertenece el ángulo
                    bin_index = int(angle / bin_width)

                    # Distribuir la magnitud en los bins adyacentes (interpolación lineal)
                    if bin_index < num_bins - 1:
                        histograms[y, x, bin_index] += magnitude * (1 - (angle % bin_width) / bin_width)
                        histograms[y, x, bin_index + 1] += magnitude * (angle % bin_width) / bin_width
                    else:
                        histograms[y, x, bin_index] += magnitude

    # Normalizar bloques de celdas
    block_size = 2  # Tamaño del bloque en celdas (por lo general, 2x2)
    block_stride = 1  # Desplazamiento del bloque en celdas (por lo general, 1 celda)
    norm_type = cv2.NORM_L2
    epsilon = 1e-4

    # Inicializar vectores de características HOG
    hog_features = []

    # Para cada bloque de celdas
    for y in range(cells_y - block_size + 1):
        for x in range(cells_x - block_size + 1):
            block = histograms[y:y + block_size, x:x + block_size, :]

            # Calcular la norma del bloque
            block_norm = np.sqrt(np.sum(block ** 2) + epsilon ** 2)

            # Normalizar el bloque y aplanar en un vector de características
            block /= block_norm
            hog_features.extend(block.flatten())
            
    return hog_features