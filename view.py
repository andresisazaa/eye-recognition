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
    histImage = cv2.calcHist([image], [0], None, [256], [150, 256])
    eyes = []
    for eye in eyesList:
        histImageDb = cv2.calcHist([eye[1]], [0], None, [256], [150, 256])
        result = np.sqrt(np.power((np.subtract(histImage, histImageDb)), 2))
        dst = distance.euclidean(histImage, histImageDb)
        eyes.append((eye[0], util.similarity_percentage(dst)))
    eyes = sorted(eyes, key=lambda x: x[1])
    eye_select = eyes[len(eyes)-1]
    return (eye_select)


def select_image():
    global imageA, imageB, imageC
    path = filedialog.askopenfilename()
    if len(path) > 0:
        frame = cv2.imread(path)
        iris = copy.copy(frame)
        pupil = processing.getPupil(frame)
        iris = processing.getIris(pupil)
        polar = processing.getPolar2CartImage(iris)
        for file in glob.glob(path_bd):
            img = cv2.imread(file)
            path = os.path.splitext(file)[0]
            eyesList.append((path, img))
        # cv2.imshow('polar', polar)

        # frame2 = cv2.imread('bd/022R_1.png')
        # iris2 = copy.copy(frame2)
        # pupil2 = getPupil(frame2)
        # # cv2.imshow('pupil', pupil2)
        # iris2 = getIris(pupil2)
        # # cv2.imshow('iris', iris2)
        # polar2 = getPolar2CartImage(iris2)
        # cv2.imwrite('bd_procesada/022R_1_polar.png', polar2)
        eye_select = comparation(polar)
        eye_percentage_select = eye_select[1]
        final_eye = cv2.imread(util.get_eye_path(eye_select[0]))
        # cv2.imshow('final', final_eye)
        # To Pil format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        polar = Image.fromarray(polar)
        final = cv2.cvtColor(final_eye, cv2.COLOR_BGR2RGB)
        final = Image.fromarray(final)
        # To Imagetk format
        frame = ImageTk.PhotoImage(frame)
        polar = ImageTk.PhotoImage(polar)
        final = ImageTk.PhotoImage(final)

        if imageA is None or imageB is None or imageC is None:
            imageA = tk.Label(image=frame)
            imageA.image = frame
            imageA.pack(side="left", padx=2, pady=2)
            # imageB = tk.Label(image=polar)
            # imageB.image = polar
            # imageB.pack(side="bottom", padx=3, pady=3)
            imageC = tk.Label(image=final)
            imageC.image = final
            imageC.pack(side="right", padx=2, pady=2)
            # lbl = tk.Label(text = f'El porcentaje de similitud es de: {eye_percentage_select} %')
            # lbl.text = f'El porcentaje de similitud es de: {eye_percentage_select} %'
            # lbl.pack(side = "top", padx="5", pady="5")
            output.set(f'{eye_percentage_select}%')
        else:
            imageA.configure(image=frame)
            imageB.configure(image=polar)
            imageC.configure(image=final)
            imageA.image = frame
            imageB.image = polar
            imageC.image = final


root = tk.Tk()
root.geometry("1500x800")
imageA = None
imageB = None
imageC = None
output = tk.StringVar()

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI

lbl = tk.Label(root, text='El porcentaje de similitud es de: ')
lbl.pack()
out = tk.Entry(root, justify="center", textvariable=output,
               state="disabled").pack()
btn = tk.Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="5", pady="5")
# btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
# kick off the GUI
root.mainloop()
cv2.waitKey(0)
cv2.destroyAllWindows()
