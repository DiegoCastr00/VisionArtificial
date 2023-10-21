import numpy as np
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
        
        
def FreemanChain(imgBinaria, conexion):
    start = primer1(imgBinaria)  # No se proporciona la función primer1, deberías reemplazarla por tu propia implementación
    print(start)
    currentP = start
    chainResult = []
    border = [start]  # Se inicia con el primer punto en el borde
    result = np.zeros_like(imgBinaria)
    result[start] = 1
    if conexion == 8:
        directions = [3, 2, 1, 
                      4,    0,
                      5, 6, 7] 
        movimiento = [(-1, -1),(-1, 0),(-1, 1),
                       (0, -1),         (0,1),
                       (1, -1),(1, 0), (1, 1)]
        direc1 = [2, 1, 0, 7, 6, 5, 4, 3]
        movs1 = [(-1, 0),(-1, 1),(0,1), (1,1),(1, 0), (1, -1),(0, -1),(-1, -1)]
        direc4 =[0, 7,6,5,4,3,2,1]
        movs4 = [(0,1), (1, 1), (1, 0),(1, -1),(0, -1),(-1, -1),(-1, 0),(-1, 1)]
        direc3 =[6, 5, 4, 3, 2, 1, 0, 7] 
        movs3 = [(1, 0),(1, -1),(0, -1),(-1, -1),(-1, 0),(-1, 1), (0,1),(1, 1)] 
        direc2 = [4, 3, 2, 1, 0, 7, 6, 5]
        movs2 = [(0, -1),(-1, -1),(-1, 0),(-1, 1), (0,1),(1, 1), (1, 0),(1, -1)]   
        pass
    elif conexion == 4:
        directions = [   1, 
                      2,    0,
                         3] 
        movimiento = [     (-1, 0),
                    (0, -1),      (0,1),
                            (1, 0)]
        direc1 = [1, 0, 3, 2]
        movs1 = [(-1,0), (0,1), (1,0), (0, -1)]
        direc4 = [0, 3, 2, 1]
        movs4 = [(0,1), (1, 0), (0, -1), (-1, 0)]
        direc3 = [3,2,1,0]
        movs3 = [(1,0), (0, -1), (-1,0), (0,1)]
        direc2 =[2,1,0, 3]
        movs2 = [(0, -1), (-1,0), (0,1), (1,0)]
        pass
    
    dicc1 = dict(zip(direc1, movs1))
    dicc2 = dict(zip(direc2, movs2))
    dicc3 = dict(zip(direc3, movs3))
    dicc4 = dict(zip(direc4, movs4))
    cuadrante = obtenerCuadrante(currentP, imgBinaria)
    diccActual = dicc1.copy()
    consecutive_empty_moves = 0
    llego = False
    while True:
        added_to_chain = False
        for direction, mov in diccActual.items():
            next_mov = (currentP[0] + mov[0], currentP[1] + mov[1])
            if 0 <= next_mov[0] < imgBinaria.shape[0] and 0 <= next_mov[1] < imgBinaria.shape[1]:
                if imgBinaria[next_mov] == 1 and result[next_mov] != 0 and next_mov == start:
                    print('forzamos termino ')
                    llego = True
                if imgBinaria[next_mov] == 1 and result[next_mov] == 0:
                    border.append(next_mov)  # Agregar el punto al borde
                    chainResult.append(direction)
                    result[next_mov] = 1  # Marcar solo el punto en el borde en el resultado
                    currentP = next_mov
                    added_to_chain = True
                    consecutive_empty_moves = 0
                    '''plt.imshow(result, cmap='binary')
                    plt.grid(True)
                    plt.xticks(np.arange(0.5, result.shape[1]+0.5, 1))
                    plt.yticks(np.arange(0.5, result.shape[0]+0.5, 1))
                    plt.show()'''
                    break
                else:
                    consecutive_empty_moves += 1
        if consecutive_empty_moves > 10 or llego:
            break
        else:
            cuadrante = obtenerCuadrante(currentP, imgBinaria)
            if cuadrante == 1:
                diccActual = dicc1
            elif cuadrante == 4:
                diccActual = dicc3
            elif cuadrante == 3:
                diccActual = dicc3
            elif cuadrante == 2:
                diccActual = dicc1
    return chainResult, border