#!/usr/bin/python

import sys
import cv2
import math
import numpy as np

def weight(f, x, y):
    Gx = (f[x+1][y] - f[x-1][y])/2
    Gy = (f[x][y+1] - f[x][y-1])/2

    d = math.sqrt(Gx*Gx + Gy*Gy)
    w = math.exp(-math.sqrt(d))
    return w

image = sys.argv[1]
img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

f = np.copy(img/255) * 1
W = np.copy(img/255) * 0
N = np.copy(img/255) * 0
g = np.copy(img/255) * 0

for x in range(1, len(f) - 1):
    for y in range(1, len(f[x]) - 1):
        W[x,y] = weight(f, x, y)

# cv2.imwrite("{}weight_py.jpg".format(image), W*255)

for x in range(1, len(f) - 1):
    for y in range(1, len(f[x]) - 1):
        for i in range(-1,2):
            for j in range(-1,2):
                N[x,y] += W[x+i,y+j]

for x in range(1, len(f) - 1):
    for y in range(1, len(f[x]) - 1):
        for i in range(-1,2):
            for j in range(-1,2):
                g[x,y] += (W[x+i,y+j]*f[x+i][y+j])/N[x,y]

G = np.array(g*255, dtype='uint8')

cv2.imwrite("{}filter_py.jpg".format(image), G)
# can1 = cv2.Canny(G, 100, 180)
# cv2.imwrite("{}canny_with.jpg".format(image), can1)
# can2 = cv2.Canny(img, 100, 180)
# cv2.imwrite("{}canny_noot.jpg".format(image), can2)
