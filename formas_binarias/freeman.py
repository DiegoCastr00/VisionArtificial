def primer1(contorno):
    for i in range(contorno.shape[0]):
        for j in range(contorno.shape[1]):
            if contorno[i, j] == 1:
                return(i, j)
        else:
            continue