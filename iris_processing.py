import cv2
import numpy as np
from scipy.spatial import distance
import utils as util
import copy

def getPupil(frame):
    dimensions = frame.shape
    pupil_img = np.zeros(dimensions, np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 45, 80, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i in contours:
        moments = cv2.moments(i)
        area = moments['m00']
        if ((area > 7000) & (area < 30000)):
            x = moments['m10']/area
            y = moments['m10']/area
            centroid = (int(x), int(y))
            cv2.drawContours(gray, i, -1, (0, 255, 0), 3)
    return (gray)


def getCircles(image):
    i = 80
    output = image.copy()
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1,
                               0.01, param1=250, param2=60, minRadius=0, maxRadius=0)
    detected_circles = np.uint16(np.around(circles))
    for (x, y, r) in detected_circles[0, :]:
        cv2.circle(output, (x, y), r, (0, 255, 0), 3)
        cv2.circle(output, (x, y), 2, (0, 255, 255), 3)
    return (circles)


def getIris(frame):
    gray = cv2.Canny(frame, 350, 5)
    gray = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_ISOLATED)
    circles = getCircles(gray)
    radius = 72
    mask = cv2.circle(frame, (384, 281), radius, (255, 255, 255), cv2.FILLED)
    mask = cv2.bitwise_not(mask)
    return (mask)


def getPolar2CartImage(image):
    height, width = image.shape
    c = (float(768/2.0), float(562/2.0))
    imgRes = cv2.warpPolar(image, (height, width), c, 260,
                           cv2.INTER_LINEAR+cv2.WARP_POLAR_LOG)
    imgRes = cv2.addWeighted(imgRes, 1.3, np.zeros(
        imgRes.shape, imgRes.dtype), 0, 0)
    return (imgRes)
