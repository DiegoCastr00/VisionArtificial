import cv2
import numpy as np
import matplotlib.pyplot as plt
def match(imagen1,descriptores1,keypoints1, imagen2,descriptores2,keypoints2, umbral=0.5,matches_thickness=2):
    bf = cv2.BFMatcher()
    coincidencias = bf.knnMatch(descriptores1, descriptores2, k=2)

    buenas_coincidencias = []
    for m, n in coincidencias:
        if m.distance < umbral* n.distance:
            buenas_coincidencias.append(m)
            
    imagen1_rgb = cv2.cvtColor(imagen1, cv2.COLOR_BGR2RGB)
    imagen2_rgb = cv2.cvtColor(imagen2, cv2.COLOR_BGR2RGB)

    keypoints1_list = [cv2.KeyPoint(float(x), float(y), float(scale)) for x, y, scale in keypoints1]
    keypoints2_list = [cv2.KeyPoint(float(x), float(y), float(scale)) for x, y, scale in keypoints2]

    def generar_color_aleatorio():
        return tuple(np.random.randint(0, 255, 3).tolist())
    
    resultado = cv2.drawMatches(
        imagen1_rgb, keypoints1_list,
        imagen2_rgb, keypoints2_list,
        buenas_coincidencias, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    for match in buenas_coincidencias:
        x1, y1 = map(lambda x: int(x), keypoints1_list[match.queryIdx].pt)
        x2, y2 = map(lambda x: int(x), keypoints2_list[match.trainIdx].pt)
    

        color = generar_color_aleatorio()
    
        cv2.line(resultado, (x1, y1), (x2 + imagen1_rgb.shape[1], y2), color, thickness=matches_thickness)
    
    
    plt.figure(figsize=(16, 16))
    plt.imshow(resultado)
    plt.title('Coincidencias SIFT')
    plt.show()
