import random

def etiquetas_c(imagen):
    def encontrar_raiz(etiqueta):
        raiz = etiqueta
        while padre[raiz] != raiz:
            raiz = padre[raiz]
        return raiz

    def union(etiqueta1, etiqueta2):
        raiz1 = encontrar_raiz(etiqueta1)
        raiz2 = encontrar_raiz(etiqueta2)
        if raiz1 != raiz2:
            padre[raiz2] = raiz1

    altura, ancho = len(imagen), len(imagen[0])
    padre = list(range(altura * ancho))
    # donde se almacenará la setiquetas
    etiquetas = [[0 for _ in range(ancho)] for _ in range(altura)]
    etiqueta_actual = 1

    for row in range(len(imagen)):
        for col in range(len(imagen[0])):
            if imagen[row][col] == 1:
                vecinos = []
                if row > 0 and etiquetas[row - 1][col] > 0:
                    vecinos.append(etiquetas[row - 1][col])
                if col > 0 and etiquetas[row][col - 1] > 0:
                    vecinos.append(etiquetas[row][col - 1])

                if not vecinos:
                    etiquetas[row][col] = etiqueta_actual
                    etiqueta_actual += 1
                else:
                    etiquetas[row][col] = min(vecinos)
                    for vecino in vecinos:
                        union(etiquetas[row][col], vecino)


    raiz_es = {}
    nueva_e = 1
    #segujda vuelta para encontrar componenetes conectados
    for row in range(altura):
        for col in range(ancho):
            if etiquetas[row][col] > 0:
                raiz = encontrar_raiz(etiquetas[row][col])
                if raiz not in raiz_es:
                    raiz_es[raiz] = nueva_e
                    nueva_e += 1
                etiquetas[row][col] = raiz_es[raiz]

    color, numcolors = contar_colores300(imagen,etiquetas)
    return color, numcolors

#Colorear las imagenes en función de las etiquetas
'''
def colores(imagen, etiquetas):
    rows, cols = len(imagen), len(imagen[0])
    imagen_color = [[(0, 0, 0) for _ in range(cols)] for _ in range(rows)]

    colores = {}
    for i in range(rows):
        for j in range(cols):
            if etiquetas[i][j] not in colores and etiquetas[i][j] != 0:
                colores[etiquetas[i][j]] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
            if etiquetas[i][j] != 0:
                imagen_color[i][j] = colores[etiquetas[i][j]]

    numero_de_colores = len(colores)
    return imagen_color, numero_de_colores'''
    

def contar_colores300(imagen, etiquetas):
    rows, cols = len(imagen), len(imagen[0])
    imagen_color = [[(0, 0, 0) for _ in range(cols)] for _ in range(rows)]

    colores = [None] * (max(max(row) for row in etiquetas) + 1)
    contador_colores = [0] * len(colores)

    for i in range(rows):
        for j in range(cols):
            etiqueta = etiquetas[i][j]
            if etiqueta != 0:
                if colores[etiqueta] is None:
                    colores[etiqueta] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )
                contador_colores[etiqueta] += 1

            if etiqueta != 0:
                imagen_color[i][j] = colores[etiqueta]

    # Filtrar colores that were assigned to more than 400 pixels
    colores_mas_de_300 = [colores[i] for i in range(len(colores)) if contador_colores[i] > 400]

    return imagen_color, colores_mas_de_300