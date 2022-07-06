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

f = np.copy(img/255)
W = np.copy(img/255) * 0
N = np.copy(img/255) * 0
g = np.copy(img/255)

libwang = ct.CDLL("./libwang.so")
libwang_wang_filter = libwang.wang_filter

libwang_wang_filter.restype = None
libwang_wang_filter.argtypes = [ndp(ct.c_double, flags="C_CONTIGUOUS"), 
                                ct.c_size_t, ct.c_size_t, 
                                ndp(ct.c_double, flags="C_CONTIGUOUS"),
                                ndp(ct.c_double, flags="C_CONTIGUOUS"),
                                ndp(ct.c_double, flags="C_CONTIGUOUS")]

libwang_wang_filter(f, f.shape[0], f.shape[1], W, N, g)

G = np.array(g*255, dtype='uint8')

cv2.imwrite("{}filter_C.jpg".format(image), G);
