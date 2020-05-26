import cv2  # Not actually necessary if you just want to create an image.
import numpy as np
import copy
import tkinter as tk
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
from matplotlib import pyplot as plt
from scipy.spatial import distance
import glob
import os
# Holds the pupil's center
centroid = (0, 0)
# Holds the iris' radius
radius = 0
# Holds the current element of the image used by the getNewEye function
currentEye = 0
# Holds the list of eyes (filenames)
eyesList = []
umbral = 100000
#####################################

def getPupil(frame):
    dimensions = frame.shape
    pupil_img = np.zeros(dimensions, np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 45,80,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i in contours:
        moments = cv2.moments(i)
        area = moments['m00']        
        if ((area > 7000) & (area < 30000)):
            print('area',area)
            x = moments['m10']/area
            y = moments['m10']/area
            global centroid
            centroid = (int(x), int(y))
            cv2.drawContours(gray,i,-1,(0,255,0),3)    
    return (gray)


def getCircles(image):
    i=80
    output = image.copy()
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT,1,0.01, param1=250, param2=60, minRadius=0, maxRadius=0)
    detected_circles = np.uint16(np.around(circles))
    print('detected circles', detected_circles.size)
    for (x,y,r) in detected_circles[0, :]:
        cv2.circle(output, (x,y), r, (0,255,0),3)
        cv2.circle(output, (x,y), 2, (0,255,255),3)
    return (circles)


# Returns the cropped image with the isolated iris and black-painted
# pupil. It uses the getCircles function in order to look for the best
# value for each image (eye) and then obtaining the iris radius in order
# to create the mask and crop.
#
# @param image		Image with black-painted pupil
# @returns image 	Image with isolated iris + black-painted pupil


def getIris(frame):
    gray= cv2.Canny(frame, 350,5)
    gray = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_ISOLATED)
    circles = getCircles(gray)    
    global radius
    radius = 72
    mask = cv2.circle(frame, (384,281), radius, (255, 255, 255), cv2.FILLED)
    mask = cv2.bitwise_not(mask)
    return (mask)

def getPolar2CartImage(image):
    height, width = image.shape
    c = (float(768/2.0), float(562/2.0))
    imgRes = cv2.warpPolar(image,(height,width),c,260,cv2.INTER_LINEAR+cv2.WARP_POLAR_LOG) #WARP_POLAR_LINEAR
    imgRes = cv2.addWeighted(imgRes, 1.3, np.zeros(imgRes.shape, imgRes.dtype), 0, 0)
    return (imgRes)

def comparation(image):
    histImage = cv2.calcHist([image], [0], None, [256], [150, 256])
    # plt.plot(histImage, color='gray')
    # plt.show()
    eyes = []
    for eye in eyesList:
        histImageDb = cv2.calcHist([eye[1]], [0], None, [256], [150, 256])
        result = np.sqrt(np.power((np.subtract(histImage,histImageDb)),2))
        dst = distance.euclidean(histImage,histImageDb)
        eyes.append((eye[0],similarity_percentage(dst)))
    eyes = sorted(eyes, key=lambda x: x[1])
    eye_select = eyes[len(eyes)-1]
    return (eye_select)

def similarity_percentage(distance):
    a = distance / umbral 
    percentage = (1-a)*100
    return (round(percentage,2)) 

def get_eye_path(polar_image_path):
    path = polar_image_path[13:19]
    return('bd/'+path+'.png')


def select_image():
    global imageA, imageB, imageC
    path = filedialog.askopenfilename()
    if len(path) > 0:
        frame = cv2.imread(path)
        iris = copy.copy(frame)
        pupil = getPupil(frame)
        iris = getIris(pupil)
        polar = getPolar2CartImage(iris)

        path_bd = 'bd_procesada/*.png'
        for file in glob.glob(path_bd):
            img = cv2.imread(file)
            path = os.path.splitext(file)[0]
            eyesList.append((path,img))
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
        print('percentage final', eye_select[1])
        # percentage = similarity_percentage(distance)
        # print('percentage', percentage)
        final_eye = cv2.imread(get_eye_path(eye_select[0]))
        cv2.imshow('final', final_eye)
        #To Pil format
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        polar = Image.fromarray(polar)
        final = cv2.cvtColor(final_eye,cv2.COLOR_BGR2RGB)
        final = Image.fromarray(final)
        #To Imagetk format
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
            mensaje = tk.Text(root,background="white", width=165, height=25)
            mensaje.pack(padx=0, pady=125)
            mensaje.insert(tk.INSERT, "Hola Mundo")#text= f'El porcentaje de similitud es de: {eye_select[1]} %'
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
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = tk.Button(root, text="Select an image", command= select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="5", pady="5")
# btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
# kick off the GUI
root.mainloop()
cv2.waitKey(0)
cv2.destroyAllWindows()


