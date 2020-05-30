''' 
Proyecto curso Procesamiento Digital de Imágenes 
Andrés Felipe Isaza Arboleda
Estefany Muriel Cano
Facultad de Ingeniería
UdeA 2019-2
'''


'''
Importanción de librerías
    OpenCV: Se utiliza para todo el tratamiento de las imagenes
    Numpy: Es una librería de estructuras de datos n-dimensinales, se utiliza como auxiliar para OpenCV
    scipy: Se utiliza una función de esta librería para calcular la distancias.
'''

import cv2
import numpy as np
from scipy.spatial import distance
import utils as util
import copy


'''
El método get_pupil se encarga de dibujar el contorno de la pupila
del ojo ingresado
'''

def get_pupil(frame):
    dimensions = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Se pasa a escala de grises
    _, thresh = cv2.threshold(gray, 45, 80, cv2.THRESH_BINARY) #Se define un umbral para obtener la máscara
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #Con base en la marcara se buscan los contornos
    for i in contours: #Este ciclo se encarga de iterar entre los contornos ingresados y dibujar los contornos en la imagen
        moments = cv2.moments(i)
        area = moments['m00']
        if ((area > 7000) & (area < 30000)): #Se encarga de evaluar donde está la información de la pupila del ojo obtenida de la máscara
            x = moments['m10']/area
            y = moments['m10']/area
            centroid = (int(x), int(y)) #Se define el centro (x,y) de la pupila con base en el area
            cv2.drawContours(gray, i, -1, (0, 255, 0), 3) #dibuja el contorno sobre la imagen
    return (gray) #Retorna el ojo en escala de grises con el borde de la pupila marcado.



'''
get_circles es una función auxiliar que nos permite usar la función de HoughCircles
para obtener el borde externo del iris
'''
def get_circles(image):
    i = 80
    output = image.copy()
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1,
                               0.01, param1=250, param2=60, minRadius=0, maxRadius=0) #Se obtienen los circulos de la imagen usando la transformada de Hough
    detected_circles = np.uint16(np.around(circles))
    for (x, y, r) in detected_circles[0, :]: #Se dibujan los circulos obtenidos en la imagen (el borde del iris)
        cv2.circle(output, (x, y), r, (0, 255, 0), 3)
        cv2.circle(output, (x, y), 2, (0, 255, 255), 3)
    return

'''
get_iris se encarga de delimitar los bordes internos y externos del iris
y dibujarlos en la imagen
'''
def get_iris(frame):
    gray = cv2.Canny(frame, 350, 5) #Se utiliza el operador Canny para dibujar deterctar bordes de la imagen
    gray = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_ISOLATED) #Se realiza un difuminado para rellenar imperfecciones pequeñas en la imagen
    get_circles(gray) # Se llama get_circles para dibujar los bordes
    radius = 72 #Se calcula el radio interior del ojo
    mask = cv2.circle(frame, (384, 281), radius, (255, 255, 255), cv2.FILLED) #Se dibuja el circulo del radio interior del ojo puila(en la imagen)
    mask = cv2.bitwise_not(mask)
    return (mask) #Se retorna la image con el iris preprocesado

'''
Esta función se encarga de hacer la transformación del iris de coordenadas cartesianas
a polares.
'''
def transform_polar2cartesian(image):
    height, width = image.shape
    c = (float(768/2.0), float(562/2.0)) #Se define el centroide que se usa como referencia para la transformación
    imgRes = cv2.warpPolar(image, (height, width), c, 260, #Se realiza el cambio de coordenadas
                           cv2.INTER_LINEAR+cv2.WARP_POLAR_LOG)
    imgRes = cv2.addWeighted(imgRes, 1.3, np.zeros(
        imgRes.shape, imgRes.dtype), 0, 0) #Se crea una imagen nueva con la transformación realizada
    return (imgRes) #Se retorna la imagen en coordenadas cartesianas
