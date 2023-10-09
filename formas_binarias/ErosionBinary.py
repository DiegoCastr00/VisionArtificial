import numpy as np

def ErosionBin(imagen_binaria, kernel):
    structE = np.ones((kernel, kernel))
    filas, columnas = imagen_binaria.shape
    filas_kernel, columnas_kernel = structE.shape
    
    resultado_erosion = np.zeros_like(imagen_binaria, dtype=np.uint8)
    
    for i in range(filas):
        for j in range(columnas):
            submatriz = imagen_binaria[i:i+filas_kernel, j:j+columnas_kernel]
            if np.array_equal(submatriz, structE):
                resultado_erosion[i+1, j+1] = 1
    
    return resultado_erosion