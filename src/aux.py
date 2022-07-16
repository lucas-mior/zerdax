from os.path import exists
import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math

i = 0
def save(img, filename, image):
    global i
    if img.save and not exists(filename):
        cv2.imwrite("{0}{1:1=2d}_{2}.png".format(img.basename, i, filename), image)
    i += 1

def savefig(img, filename, fig):
    global i
    fig.savefig("{0}{1:1=2d}_{2}.png".format(img.basename, i, filename))
    i += 1

def radius(x1,y1,x2,y2):
    return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def theta(x1,y1,x2,y2):
    return math.degrees(math.atan2((y1-y2),(x2-x1)))
