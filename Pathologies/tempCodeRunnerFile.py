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