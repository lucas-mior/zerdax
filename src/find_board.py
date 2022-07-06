import cv2
import numpy as np

import ctypes as ct
from numpy.ctypeslib import ndpointer as ndp

def wang_filter(img):
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
    return G

def find_board(img):

    small_wang = wang_filter(img.small)
    small_gaus = cv2.GaussianBlur(img.small, (3,3), 0)

    canny_wang = cv2.Canny(small_wang, 100, 180)
    canny_gaus = cv2.Canny(small_gaus, 100, 180)

    cv2.imwrite("{}canny_wang.jpg".format(img.basename), canny_wang)
    cv2.imwrite("{}canny_gaus.jpg".format(img.basename), canny_gaus)

    rho = 1             # distance resolution in pixels of the Hough grid
    theta = np.pi / 180 # angular resolution in radians of the Hough grid
    threshold = 15      # minimum number of votes (intersections in Hough grid cell)

    lines_wang = cv2.HoughLinesP(canny_wang, rho, theta, threshold)
    lines_gaus = cv2.HoughLinesP(canny_gaus, rho, theta, threshold)

    line_image_wang = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR) * 0
    line_image_gaus = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR) * 0
    gray3ch = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR)

    for line in lines_wang:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image_wang,(x1,y1),(x2,y2),(0,0,250),2)

    for line in lines_gaus:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image_gaus,(x1,y1),(x2,y2),(0,0,250),2)

    hough_wang = cv2.addWeighted(gray3ch, 1, line_image_wang, 0.8, 0)
    hough_gaus = cv2.addWeighted(gray3ch, 1, line_image_gaus, 0.8, 0)

    cv2.imwrite("{}hough_wang.jpg".format(img.basename), hough_wang)
    cv2.imwrite("{}hough_gaus.jpg".format(img.basename), hough_gaus)

    # ret, cvthr = cv2.threshold(img.gray, 160, 255, cv2.THRESH_BINARY)
    return (10, 300, 110, 310)
