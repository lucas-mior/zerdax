#!/usr/bin/python

import sys
import ctypes as ct
from numpy.ctypeslib import ndpointer
import numpy as np
import sys
import cv2
import math
import time

image = sys.argv[1]
img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

W = np.copy(img/255) * 0
filt = np.copy(img) * 0

fimg = np.copy(img/255)

DOUBLEPtr = ct.POINTER(ct.c_double)
DOUBLEPtrPtr = ct.POINTER(DOUBLEPtr)
libwang = ct.CDLL("./libwang.so")
libwang_weight_array = libwang.weight_array

libwang_weight_array.restype = None
libwang_weight_array.argtypes = [ndpointer(ct.c_double, flags="C_CONTIGUOUS"), ct.c_size_t, ct.c_size_t, ndpointer(ct.c_double, flags="C_CONTIGUOUS")]

libwang_weight_array(fimg, fimg.shape[0], fimg.shape[1], W)
W = W.reshape(fimg.shape)

print("W :",W.shape)
cv2.imwrite("{}weight_C.jpg".format(image), W*255);
print("W[200:200] =", W[200,200])

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
                filt[x,y] += (W[x+i,y+j]*img[x+i][y+j])/N[x,y]

cv2.imwrite("{}filt.jpg".format(image), filt)
can1 = cv2.Canny(filt, 100, 180)
cv2.imwrite("{}canny_with.jpg".format(image), can1)
can2 = cv2.Canny(img, 100, 180)
cv2.imwrite("{}canny_noot.jpg".format(image), can2)
