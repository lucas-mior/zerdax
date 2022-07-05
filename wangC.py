#!/usr/bin/python

import sys
import ctypes as ct
from numpy.ctypeslib import ndpointer as ndp
import numpy as np
import sys
import cv2
import math

image = sys.argv[1]
img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

W = np.copy(img/255) * 0
N = np.copy(img/255) * 0
g = np.copy(img/255) * 0

fimg = np.copy(img/255)

libwang = ct.CDLL("./libwang.so")
libwang_wang_filter = libwang.wang_filter

libwang_wang_filter.restype = None
libwang_wang_filter.argtypes = [ndp(ct.c_double, flags="C_CONTIGUOUS"), 
                                ct.c_size_t, ct.c_size_t, 
                                ndp(ct.c_double, flags="C_CONTIGUOUS"),
                                ndp(ct.c_double, flags="C_CONTIGUOUS"),
                                ndp(ct.c_double, flags="C_CONTIGUOUS")]

libwang_wang_filter(fimg, fimg.shape[0], fimg.shape[1], W, N, g)
print(g)
cv2.imwrite("{}filt_C.jpg".format(image), g);

exit()

# W = W.reshape(fimg.shape[0], fimg.shape[1], order='F')

# cv2.imwrite("{}weight_C.jpg".format(image), W*255);

N = np.copy(W) * 0

for x in range(1, len(img) - 1):
    for y in range(1, len(img[x]) - 1):
        for i in range(-1,2):
            for j in range(-1,2):
                N[x,y] += W[x+i,y+j]

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
