'''
canalVerde = np.zeros_like(bordes_detectados)
canalRojo = np.where(bordes_detectados == 1, 255, bordes_detectados)
CanalAzul = np.zeros_like(bordes_detectados)
matriz_visualizada = np.stack((canalVerde, canalRojo, CanalAzul), axis=-1)
plt.imshow(matriz_visualizada)
plt.title('Bordes Detectados en Imagen Binaria')
plt.show()


combination = np.zeros_like(matriz_visualizada)  # Crea una matriz de ceros del mismo tamaño que globalX

for x in range(img.shape[1]):
    for y in range(img.shape[0]):
        if globalXI[y, x] == 1:
            combination[y, x, :] =[255,255,255]
        else:
            combination[y, x, :] =[0,0,0]
        if (matriz_visualizada[y,x,1] == 255):
            combination[y, x, :] = matriz_visualizada[y,x,:]

plt.imshow(combination)
plt.show()'''