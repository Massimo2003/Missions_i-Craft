import cv2
import matplotlib.pyplot as plt
import numpy as np

def loadImage(path, colormap):
    img = cv2.imread(path)
    if colormap == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif colormap == "color":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print("invalid colormap")
        
imgColor = loadImage("AllImages/Desquamation3.jpg", "color")
imgColorDraw = loadImage("AllImages/Desquamation3.jpg", "color")
imgGray = loadImage("AllImages/Desquamation3.jpg", "gray")

def binarisation(i, type, inv):
    if type == "otsu" and inv == "norm":
        return cv2.threshold(i, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if type == "otsu" and inv == "inv":
        return cv2.threshold(i, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    if type == "tri" and inv == "norm":
        return cv2.threshold(i, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    if type == "tri" and inv == "inv":
        return cv2.threshold(i, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_TRIANGLE)
    
    else:
        print("Invalid parameters")
        
retOtsu, threshOtsu = binarisation(imgGray, "otsu", "norm")
retTriangle, threshTriangle = binarisation(imgGray, "tri", "norm" )
retTriangle, threshTriangleInv = binarisation(imgGray, "tri", "inv" )
print(retOtsu)
print(retTriangle)

def contoursFill(ibn, i, clrCnt, clrfill):
    contours, hierarchy = cv2.findContours(ibn, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = max(contours, key = cv2.contourArea)
    imgCnt = cv2.drawContours(i, contours, -1, clrCnt, 3)
    return cv2.fillPoly(imgCnt, pts=contours, color=clrfill)

imgCntFill = contoursFill(threshTriangleInv, imgColorDraw, (0, 0, 255), (0,100,255))

def plotImage(title, i):
    plt.figure(title)
    plt.imshow(i, cmap="gray")
    return plt.show()

plotImage("Color", imgColor)
plotImage("Gray", imgGray)
plotImage("Binaire triangle", threshTriangle)
plotImage("Binaire triangle inverse", threshTriangleInv)
plotImage("Contours filling", imgCntFill)