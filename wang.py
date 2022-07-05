#!/usr/bin/python

import sys
import cv2
import math
import numpy as np

def weight(f, x,y):
    f = f/255
    Gx = (f[x+1][y] - f[x-1][y])/2
    Gy = (f[x][y+1] - f[x][y-1])/2

    d = math.sqrt(Gx*Gx + Gy*Gy)
    w = math.exp(-math.sqrt(d))
    return w

image = sys.argv[1]
img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

W = np.copy(img/255) * 0
out = np.copy(img) * 0

print(np.min(img))
print(np.max(img))

for x in range(1, len(img) - 1):
    for y in range(1, len(img[x]) - 1):
        W[x,y] = weight(img, x, y)

print("W[200,200] =", W[200,200])

cv2.imwrite("{}weight_py.jpg".format(image), W*255)
exit()

N = np.copy(W) * 0

for x in range(1, len(img) - 1):
    for y in range(1, len(img[x]) - 1):
        for i in range(-1,2):
            for j in range(-1,2):
                N[x,y] += W[x+i,y+j]
                if W[x+i,y+j] > 0.99:
                    print("N[x,y] = {} + W[x+{},y+{}], {}".format(N[x,y], i, j, W[x+i,y+j]))

for x in range(1, len(img) - 1):
    for y in range(1, len(img[x]) - 1):
        for i in range(-1,2):
            for j in range(-1,2):
                out[x,y] += (W[x+i,y+j]*img[x+i][y+j])/N[x,y]


cv2.imwrite("{}out.jpg".format(image), out)
can1 = cv2.Canny(out, 100, 180)
cv2.imwrite("{}canny_with.jpg".format(image), can1)
can2 = cv2.Canny(img, 100, 180)
cv2.imwrite("{}canny_noot.jpg".format(image), can2)
