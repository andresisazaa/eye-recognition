import cv2
import numpy as np
import copy
import glob
import os
import tkinter as tk
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
from matplotlib import pyplot as plt
from scipy.spatial import distance
import iris_processing as processing
import utils as util

eyesList = []
eye_percentage_select = 0
path_bd = 'bd_procesada/*.png'
#####################################


def comparation(image):
    #Se saca el histograma de la imagen que escogió el usuario
    histImage = cv2.calcHist([image], [0], None, [256], [150, 256])
    eyes = [] #Definimos un arreglo para guardar los histogramas
    for eye in eyesList:
        #Sacamos el histograma de cada ojo con el que vamos a comparar
        histImageDb = cv2.calcHist([eye[1]], [0], None, [256], [150, 256])
        #Calculamos la distancia euclidiana entre el histograma de la imagen a comparar y la de la base de datos
        dst = distance.euclidean(histImage, histImageDb)
        #Guardamos el path del ojo y su respectivo porcentaje de similitud
        eyes.append((eye[0], util.similarity_percentage(dst)))
    #Ordenamos de menor a mayor
    eyes = sorted(eyes, key=lambda x: x[1])
    #Tomamos el mejor porcentaje
    eye_select = eyes[len(eyes)-1]
    return (eye_select)

#En este método es para mostrar la interfaz visual y relacionarla con la lógica previamente definida para la idenntificación de irirs
def select_image():
    global imageA, imageB, imageC #Definimos unas variables globales que van a alojar las imágenes que mostraremos en la vista
    path = filedialog.askopenfilename() #le asignamos a la variable path la ruta de la imagés seleccionada por el usuario
    if len(path) > 0: #Si seleccionaron una imagen entramos a ejecutar los métodos de reconocimiento
        frame = cv2.imread(path) #Asignamos a frame la variable seleccionada
        iris = copy.copy(frame) #Hacemos una copia de la imagen anterior
        pupil = processing.getPupil(frame) #Obtenemos la pupila del ojo
        iris = processing.getIris(pupil) #Obtenemos el iris del ojo sin la pupila
        polar = processing.getPolar2CartImage(iris) # Pasamos el círculo obtenido a coordenadas cartesianas
        #Se hace un ciclo para obtener cada imagen de la base de datos ya previamente procesadas y compararlas con la nueva entrada
        for file in glob.glob(path_bd): 
            img = cv2.imread(file) #se asigna a img la imagen extraida de la base de datos
            path = os.path.splitext(file)[0] #Se extrae el nombre del archivo
            eyesList.append((path, img)) #Ee le pasa a este arreglo la imagen y la ruta del archivo para tener un registro
        eye_select = comparation(polar) #Aquí se pasa la imagen que ingresó el usuario en coordenadas cartesianas para compararla con los otros ojos
        eye_percentage_select = eye_select[1] #Obtenemos el mejor porcentaje  
        final_eye = cv2.imread(util.get_eye_path(eye_select[0])) #Buscamos la foto del ojo con la mejor coincidencia 
       #To Pil format
        dimensions = (500,400) #Definimos las nuevas dimensiones de las imágenes 
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #Volvemos la imagen a su color original
        #Le cambiamos las dimensiones a las imágenes ya que son muy grandes para mostrarlas en la aplicación
        frame = cv2.resize(frame, dsize=dimensions, interpolation=cv2.INTER_CUBIC)
        frame = Image.fromarray(frame)
        polar = cv2.resize(polar, dsize=(400,400), interpolation=cv2.INTER_CUBIC)
        polar = polar[0:400, 350:400]
        polar = cv2.resize(polar, dsize=(100,400), interpolation=cv2.INTER_CUBIC)
        polar = Image.fromarray(polar)
        final = cv2.cvtColor(final_eye,cv2.COLOR_BGR2RGB)
        final = cv2.resize(final, dsize=dimensions, interpolation=cv2.INTER_CUBIC)
        final = Image.fromarray(final)
        #To Imagetk format
        frame = ImageTk.PhotoImage(frame)
        polar = ImageTk.PhotoImage(polar)
        final = ImageTk.PhotoImage(final) 

        if imageA is None or imageB is None or imageC is None:
            #Configuramos los camppos para mostrar las imágenes y el texto
            imageA = tk.Label(image=frame)
            imageA.image = frame
            imageA.pack(side="left", padx=1, pady=1)
            imageB = tk.Label(image=polar)
            imageB.image = polar
            imageB.pack(side="left", padx=1, pady=1)
            imageC = tk.Label(image=final)
            imageC.image = final
            imageC.pack(side="right", padx=2, pady=2)
            output.set(f'{eye_percentage_select}%')
        else:
            imageA.configure(image=frame)
            imageB.configure(image=polar)
            imageC.configure(image=final)
            imageA.image = frame
            imageB.image = polar
            imageC.image = final
            output.set(f'{eye_percentage_select}%')


root = tk.Tk()
root.geometry("1200x700")
imageA = None
imageB = None
imageC = None
output = tk.StringVar()


lbl = tk.Label(root, text='El porcentaje de similitud es de: ')
lbl.pack()
out = tk.Entry(root, justify="center", textvariable=output,
               state="disabled").pack()
btn = tk.Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", padx="5", pady="5")
# kick off the GUI
root.mainloop()
cv2.waitKey(0)
cv2.destroyAllWindows()
