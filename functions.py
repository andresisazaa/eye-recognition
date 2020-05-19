import cv2  # Not actually necessary if you just want to create an image.
import numpy as np
import copy
# Holds the pupil's center
centroid = (0, 0)
# Holds the iris' radius
radius = 0
# Holds the current element of the image used by the getNewEye function
currentEye = 0
# Holds the list of eyes (filenames)
eyesList = []
#####################################


# blank_image = np.zeros((height,width,3), np.uint8)

# img = np.ones((300, 300, 1), np.uint8)*255
# cv2.imshow('image', img)
# print(np.shape(img))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def getPupil(frame):
    dimensions = frame.shape
    print(dimensions)
    pupil_img = np.zeros(dimensions, np.uint8)
    cv2.imshow('frame', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 45,80,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print('Number contours'+ str(len(contours)))
    
    # cv2.imshow('img',frame)
    cv2.imshow('imgray',gray)
    for i in contours:
        moments = cv2.moments(i)
        area = moments['m00']        
        if ((area > 7000) & (area < 30000)):
            print('area',area)
            pupilArea = area
            x = moments['m10']/area
            y = moments['m10']/area
            global centroid
            centroid = (int(x), int(y))
            cv2.drawContours(gray,i,-1,(0,255,0),3)    
      
    cv2.imshow('img',gray)    
    
    
    return (pupil_img)


def getCircles(image):
    height, width = image.shape
    print(height, width)
    i = 80
    while i < 350:
        storage = np.array((width, 1), np.float32)
        storage = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT,2, 100, 30, i)
        circles = np.asarray(storage)
        print(f'Circlessss {circles}')
        if (len(circles) == 1):
            return circles
        i += 1
    return ([])


# Returns the cropped image with the isolated iris and black-painted
# pupil. It uses the getCircles function in order to look for the best
# value for each image (eye) and then obtaining the iris radius in order
# to create the mask and crop.
#
# @param image		Image with black-painted pupil
# @returns image 	Image with isolated iris + black-painted pupil


def getIris(frame):
    iris = []
    copyImg = frame.copy()
    resImg = frame.copy()
    height, width, channels = frame.shape
    # grayImg = cv2.CreateImage(cv2.GetSize(frame), 8, 1)
    grayImg = np.array((height, width, channels), np.uint8)
    # mask = cv2.CreateImage(cv2.GetSize(frame), 8, 1)
    mask = np.array((height, width, channels), np.uint8)
    # storage = cv2.CreateMat(frame.width, 1, cv2.CV_32FC3)
    print(frame)
    storage = np.array((width, 1), np.float32)
    # cv2.CvtColor(frame, grayImg, cv2.CV_BGR2GRAY)
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.Canny(grayImg, grayImg, 5, 70, 3)
    grayImg = cv2.Canny(grayImg, 350,5)
    #cv2.imshow('imagen que le pasa a canny', grayImg)
    # cv2.Smooth(grayImg, grayImg, cv2.CV_GAUSSIAN, 7, 7)
    grayImg = cv2.GaussianBlur(grayImg, (1, 1), cv2.BORDER_ISOLATED)
    #print(f'GRAY IMG {grayImg}')
    cv2.imshow('imagen que le pasa a circles', grayImg)
    circles = getCircles(grayImg)
    iris.append(resImg)
    for circle in circles:
        rad = int(circle[0][2])
        global radius
        radius = rad
        # cv2.Circle(mask, centroid, rad, cv2.CV_RGB(
        #     255, 255, 255), cv2.CV_FILLED)
        cv2.circle(mask, centroid, rad, (255, 0, 0), 2)
        mask = cv2.bitwise_not(mask)
        # cv2.Sub(frame, copyImg, resImg, mask)
        resImg = cv2.subtract(frame, copyImg, mask)
        x = int(centroid[0] - rad)
        y = int(centroid[1] - rad)
        w = int(rad * 2)
        h = w
        # cv2.SetImageROI(resImg, (x, y, w, h)) ************
        resImg = resImg[y:y+h, x:x+w]
        # cropImg = cv2.CreateImage((w, h), 8, 3)
        cropImg = np.array((w, h), np.uint8)
        cropImg = resImg.copy()  # cv2.Copy(resImg, cropImg)
        # cv2.ResetImageROI(resImg) *******************
        resImg = resImg[y:y-h, x:x-w]
        return(cropImg)
    return (resImg)


frame = cv2.imread("022L_3.png")
# print(frame)
iris = copy.copy(frame)
pupil = getPupil(frame)
# print(f'FRAME {frame}')
# print(f'PUPIL {pupil}')
# iris = getIris(pupil)
# cv2.imshow('pupil', pupil)
# cv2.imshow('iris', iris)
cv2.waitKey(0)
cv2.destroyAllWindows()
