def obtenerCuadrante(punto, image):
    ancho_total = image.shape[1]
    alto_total = image.shape[0]
    x, y = punto[1], punto[0]  
    if x < ancho_total / 2 and y < alto_total / 2:
        #superior izquierda
        return 2
    elif x >= ancho_total / 2 and y < alto_total / 2:
        #superior derecha
        return 1
    elif x < ancho_total / 2 and y >= alto_total / 2:
        #inferior izquierda
        return 3
    elif x >= ancho_total / 2 and y >= alto_total / 2:
        #inferior derecha
        return 4
    else:
        return "No se encuentra en ningún cuadrante"


def primer1(contorno):
    for i in range(contorno.shape[0]):
        for j in range(contorno.shape[1]):
            if contorno[i, j] == 1:
                return(i, j)
        else:
            continue