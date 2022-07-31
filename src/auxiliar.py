import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np

i = 0

def determinant(a, b):
    return a[0]*b[1] - a[1]*b[0]

def save(img, filename, image):
    global i
    # cv2.imwrite("{}{:02d}_{}.png".format(img.basename, i, filename), image)
    i += 1

def savefig(img, filename, fig):
    global i
    fig.savefig("{}{:02d}_{}.png".format(img.basename, i, filename))
    i += 1

def radius(x1,y1,x2,y2):
    return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def theta(x1,y1,x2,y2):
    orientation = math.atan2(y1-y2, x2-x1)
    orientation = math.degrees(orientation)
    return orientation

def radius_theta(lines):
    dummy = np.zeros((lines.shape[0], 1, 6), dtype='int32')
    dummy[:,0,0:4] = np.copy(lines[:,0,0:4])
    lines = dummy
    i = 0
    for line in lines:
        for x1,y1,x2,y2,r,t in line:
            lines[i, 0, 4] = round(radius(x1,y1,x2,y2))
            lines[i, 0, 5] = round(theta(x1,y1,x2,y2))
            i += 1
    return lines
