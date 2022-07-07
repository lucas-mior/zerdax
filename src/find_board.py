import cv2
import numpy as np
import math

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import ctypes as ct
from numpy.ctypeslib import ndpointer as ndp

def radius(x1,y1,x2,y2):
    return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def theta(x1,y1,x2,y2):
    return math.degrees(math.atan2((y2-y1),(x2-x1)))

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

    small_wang = wang_filter(img.gray)
    small_gaus = cv2.GaussianBlur(img.gray, (3,3), 0)

    canny_wang = cv2.Canny(small_wang, 80, 170)
    canny_gaus = cv2.Canny(small_gaus, 80, 170)

    cv2.imwrite("test/{}1canny_wang.jpg".format(img.basename), canny_wang)
    cv2.imwrite("test/{}1canny_gaus.jpg".format(img.basename), canny_gaus)

    lines_wang = cv2.HoughLinesP(canny_wang, 1, np.pi / 180, 70,        None, 75,        15)
    lines_gaus = cv2.HoughLinesP(canny_gaus, 1, np.pi / 180, 70,        None, 75,        15)
                   # HoughLinesP(image,    RHo,       theta, threshold, lines, minLength, maxGap)

    # dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
    # rho : The resolution of the parameter r in pixels. We use 1 pixel.
    # theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
    # threshold: The minimum number of intersections to "*detect*" a line
    # lines: A vector that will store the parameters (xstart,ystart,xend,yend) of the detected lines
    # minLineLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
    # maxLineGap: The maximum gap between two points to be considered in the same line.

    ## draw hough
    line_image_wang = cv2.cvtColor(img.gray, cv2.COLOR_GRAY2BGR) * 0
    line_image_gaus = cv2.cvtColor(img.gray, cv2.COLOR_GRAY2BGR) * 0
    gray3ch = cv2.cvtColor(img.gray, cv2.COLOR_GRAY2BGR)

    gray3ch_canny_wang = cv2.cvtColor(canny_wang, cv2.COLOR_GRAY2BGR)
    gray3ch_canny_gaus = cv2.cvtColor(canny_gaus, cv2.COLOR_GRAY2BGR)

    # lines_wang = lines_wang / img.fact
    # lines_wang = lines_wang.astype(int)
    # lines_gaus = lines_gaus / img.fact
    # lines_gaus = lines_gaus.astype(int)

    i = 0
    line_wang_polar = np.empty((lines_wang.shape[0], 1, 2))
    for line in lines_wang:
        for x1,y1,x2,y2 in line:
            line_wang_polar[i] = (radius(x1,y1,x2,y2), theta(x1,y1,x2,y2))
            cv2.line(line_image_wang,(x1,y1),(x2,y2),(0,0,250), round(2/img.fact))
            i += 1

    i = 0
    line_gaus_polar = np.empty((lines_gaus.shape[0], 1, 2))
    for line in lines_gaus:
        for x1,y1,x2,y2 in line:
            line_gaus_polar[i] = (radius(x1,y1,x2,y2), theta(x1,y1,x2,y2))
            cv2.line(line_image_gaus,(x1,y1),(x2,y2),(0,0,255), round(2/img.fact))
            i += 1

    hough_wang = cv2.addWeighted(gray3ch, 0.5, line_image_wang, 0.8, 0)
    hough_gaus = cv2.addWeighted(gray3ch, 0.5, line_image_gaus, 0.8, 0)

    cv2.imwrite("test/{}2hough_wang.jpg".format(img.basename), hough_wang)
    cv2.imwrite("test/{}2hough_gaus.jpg".format(img.basename), hough_gaus)

    hough_on_canny_wang = cv2.addWeighted(cv2.bitwise_not(gray3ch_canny_wang), 0.2, line_image_wang, 0.8, 0)
    hough_on_canny_gaus = cv2.addWeighted(cv2.bitwise_not(gray3ch_canny_gaus), 0.2, line_image_gaus, 0.8, 0)

    cv2.imwrite("test/{}3hough_on_canny_wang.jpg".format(img.basename), hough_on_canny_wang)
    cv2.imwrite("test/{}3hough_on_canny_gaus.jpg".format(img.basename), hough_on_canny_gaus)

    # index = lines_wang[:,0,0].argmax()
    # lin = lines_wang[index, 0, :]

    fig_wang = plt.figure()
    ax = fig_wang.add_subplot(111, xlabel='angle', ylabel='radius', xlim=(-90, 90))
    ax.plot(line_wang_polar[:,0,1], line_wang_polar[:,0,0], linestyle='', marker='.', color='blue', label='wang', alpha=0.8)
    ax.plot(line_gaus_polar[:,0,1], line_gaus_polar[:,0,0], linestyle='', marker='.', color='red', label='gauss', alpha=0.8)
    ax.legend()
    fig_wang.savefig('test/{}3_polar.png'.format(img.basename))

    # ret, cvthr = cv2.threshold(img.gray, 160, 255, cv2.THRESH_BINARY)
    return (10, 300, 110, 310)
