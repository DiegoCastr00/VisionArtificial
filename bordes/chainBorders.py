import numpy as np
import matplotlib.pyplot as plt

def bettle(inicio, contorno):
    visited = np.zeros_like(contorno)
    contour_list = [inicio]
    xi = inicio[1]
    yi = inicio[0]
    visited[yi, xi] = 1
    initial = contorno[yi, xi]

    while True:
        direction_found = False
        for direction in range(1, 9):
            x_next = xi + (direction % 3) - 1
            y_next = yi + (direction // 3) - 1
            if 0 <= x_next < contorno.shape[1] and 0 <= y_next < contorno.shape[0]:
                if contorno[y_next, x_next] == initial and visited[y_next, x_next] == 0:
                    contour_list.append((y_next, x_next))
                    visited[y_next, x_next] = 1
                    xi = x_next
                    yi = y_next
                    direction_found = True
                    break
        if not direction_found or contour_list[-1] == inicio:
            break
    return contour_list

def primer1(contorno):
    for i in range(contorno.shape[0]):
        for j in range(contorno.shape[1]):
            if contorno[i, j] == 1:
                return(i, j)
        else:
            continue
            
def replaceColor(image, listContorno, RGB):
    res = np.copy(image)
    for point in listContorno:
        y, x = point[0], point[1] # Asegurarse de que point sea un objeto iterable
        res[y, x, :] = [RGB[0], RGB[1], RGB[2]]
    plt.imshow(res)
