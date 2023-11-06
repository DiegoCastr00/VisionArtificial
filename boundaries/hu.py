import numpy as np
# Función para calcular momentos centrales
def central_moments(image, p, q):
    height, width = image.shape
    m00 = np.sum(image)
    m10 = np.sum(np.array([[x * image[y, x] for x in range(width)] for y in range(height)]))
    m01 = np.sum(np.array([[y * image[y, x] for x in range(width)] for y in range(height)]))
    x_bar = m10 / m00
    y_bar = m01 / m00
    central_moment = np.sum(np.array([[(x - x_bar) ** p * (y - y_bar) ** q * image[y, x] for x in range(width)] for y in range(height)]))
    return central_moment


def hu(image):
    # Calcular los momentos centrales
    m00 = central_moments(image, 0, 0)
    m20 = central_moments(image, 2, 0)
    m02 = central_moments(image, 0, 2)
    m11 = central_moments(image, 1, 1)
    m30 = central_moments(image, 3, 0)
    m03 = central_moments(image, 0, 3)
    m12 = central_moments(image, 1, 2)
    m21 = central_moments(image, 2, 1)

    # Normalizar los momentos por m00
    phi1 = m20 / (m00 ** 2) + m02 / (m00 ** 2)
    phi2 = ((m20 / (m00 ** 2)) - (m02 / (m00 ** 2))) ** 2 + (4 * m11 / (m00 ** 2)) ** 2
    phi3 = ((m30 / (m00 ** 2)) - (3 * m12 / (m00 ** 2))) ** 2 + ((3 * m21 / (m00 ** 2)) - (m03 / (m00 ** 2))) ** 2
    phi4 = ((m30 / (m00 ** 2)) + (m12 / (m00 ** 2))) ** 2 + ((m21 / (m00 ** 2)) + (m03 / (m00 ** 2))) ** 2
    phi5 = ((m30 / (m00 ** 2)) - (3 * m12 / (m00 ** 2))) * ((m30 / (m00 ** 2)) + (m12 / (m00 ** 2))) * (((m30 / (m00 ** 2)) + (m12 / (m00 ** 2))) ** 2 - (3 * ((m21 / (m00 ** 2)) + (m03 / (m00 ** 2))) ** 2)) + ((3 * m21 / (m00 ** 2)) - (m03 / (m00 ** 2))) * ((m21 / (m00 ** 2)) + (m03 / (m00 ** 2))) * (3 * (((m30 / (m00 ** 2)) + (m12 / (m00 ** 2))) ** 2) - (((m21 / (m00 ** 2)) + (m03 / (m00 ** 2))) ** 2))
    phi6 = ((m20 / (m00 ** 2)) - (m02 / (m00 ** 2))) * (((m30 / (m00 ** 2)) + (m12 / (m00 ** 2))) ** 2 - (((m21 / (m00 ** 2)) + (m03 / (m00 ** 2))) ** 2)) + (4 * m11 / (m00 ** 2)) * ((m30 / (m00 ** 2)) + (m12 / (m00 ** 2))) * ((m21 / (m00 ** 2)) + (m03 / (m00 ** 2)))
    phi7 = ((3 * m21 / (m00 ** 2)) - (m03 / (m00 ** 2))) * ((m30 / (m00 ** 2)) + (m12 / (m00 ** 2))) * (((m30 / (m00 ** 2)) + (m12 / (m00 ** 2))) ** 2 - (3 * ((m21 / (m00 ** 2)) + (m03 / (m00 ** 2))) ** 2)) - ((m30 / (m00 ** 2)) - (3 * m12 / (m00 ** 2))) * ((m21 / (m00 ** 2)) + (m03 / (m00 ** 2))) * (3 * (((m30 / (m00 ** 2)) + (m12 / (m00 ** 2))) ** 2) - (((m21 / (m00 ** 2)) + (m03 / (m00 ** 2))) ** 2))

    return phi1, phi2, phi3, phi4, phi5, phi6, phi7