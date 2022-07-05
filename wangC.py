#!/usr/bin/python

import sys
import ctypes as ct
from numpy.ctypeslib import ndpointer
import numpy as np
import sys
import cv2
import math

def pyweight(f, x,y):
    Gx = (f[x+1][y] - f[x-1][y])/2
    Gy = (f[x][y+1] - f[x][y-1])/2

    d = math.sqrt(Gx*Gx + Gy*Gy)
    w = math.exp(-math.sqrt(d))
    return w

image = sys.argv[1]
img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

W = np.copy(img/255) * 0
Wp = np.copy(W)
out = np.copy(img) * 0

print(np.min(img))
print(np.max(img))

fimg = img / 255
fimg = np.copy(fimg)

DOUBLEPtr = ct.POINTER(ct.c_double)
DOUBLEPtrPtr = ct.POINTER(DOUBLEPtr)
libwang = ct.CDLL("./libwang.so")
libwang_weight = libwang.weight
libwang_weight.argtypes = [DOUBLEPtrPtr]
libwang_weight.restype = ct.c_double
# weight.restype = ndpointer(dtype=c_float,shape=(1))

# np_arr_2d = np.empty([10, 10], dtype=np.double)
np_arr_2d = np.copy(fimg);

# magic
ct_arr = np.ctypeslib.as_ctypes(np_arr_2d)
DOUBLEPtrArr = DOUBLEPtr * ct_arr._length_
ct_ptr = ct.cast(DOUBLEPtrArr(*(ct.cast(row, DOUBLEPtr) for row in ct_arr)), DOUBLEPtrPtr)

for x in range(1, len(fimg) - 1):
    for y in range(1, len(fimg[x]) - 1):
        W[x,y] = libwang_weight(ct_ptr, x, y)
        Wp[x,y] = pyweight(np_arr_2d, x, y)
        print("W = {}, Wp = {}".format(W[x,y],Wp[x,y]))

print(W)
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

# print("\n{0:s} returned: {1:f}".format(libwang_weight.__name__,res))
