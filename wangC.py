#!/usr/bin/python

import sys
import ctypes as ct
from numpy.ctypeslib import ndpointer
import numpy as np
import sys
import cv2
import math

image = sys.argv[1]
img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

W = np.copy(img/255) * 0
filt = np.copy(img) * 0

fimg = img / 255

DOUBLEPtr = ct.POINTER(ct.c_double)
DOUBLEPtrPtr = ct.POINTER(DOUBLEPtr)
libwang = ct.CDLL("./libwang.so")
libwang_weight_array = libwang.weight_array
libwang_weight_array.argtypes = [DOUBLEPtrPtr]
# libwang_weight_array.restype = ct.c_double
libwang_weight_array.restype = ndpointer(dtype = ct.c_double, shape = fimg.shape)

# np_arr_2d = np.empty([10, 10], dtype=np.double)
np_arr_2d = np.copy(fimg);

# magic
fimg_c = np.ctypeslib.as_ctypes(np_arr_2d)
DOUBLEPtrArr = DOUBLEPtr * fimg_c._length_
fimg_ptr = ct.cast(DOUBLEPtrArr(*(ct.cast(row, DOUBLEPtr) for row in fimg_c)), DOUBLEPtrPtr)

W = np.ctypeslib.as_array(libwang_weight_array(fimg_ptr, fimg.shape[0], fimg.shape[1]), fimg.shape)
W = 255*W

print(W)
cv2.imwrite("teste.jpg", W);

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
