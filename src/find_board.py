import cv2
import numpy as np
import math

from Image import Image
import glob

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import ctypes as ct
from numpy.ctypeslib import ndpointer as ndp

def radius(x1,y1,x2,y2):
    return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def theta(x1,y1,x2,y2):
    return math.degrees(math.atan2((y2-y1),(x2-x1)))

def find_straight_lines(img_wang):
    canny_wang = cv2.Canny(img_wang, 80, 170)

    # cv2.imwrite("test/{}1canny_wang.jpg".format(img.basename), canny_wang)

    lines_wang = cv2.HoughLinesP(canny_wang, 1, np.pi / 180, 70,        None, 75,        15)
                   # HoughLinesP(image,    RHo,       theta, threshold, lines, minLength, maxGap)
    # dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
    # rho : The resolution of the parameter r in pixels. We use 1 pixel.
    # theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
    # threshold: The minimum number of intersections to "*detect*" a line
    # lines: A vector that will store the parameters (xstart,ystart,xend,yend) of the detected lines
    # minLineLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
    # maxLineGap: The maximum gap between two points to be considered in the same line.

    return lines_wang

def draw_hough(lines_wang):
    ## draw hough
    line_image_wang = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR) * 0
    gray3ch = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR)

    gray3ch_canny_wang = cv2.cvtColor(canny_wang, cv2.COLOR_GRAY2BGR)

    # lines_wang = lines_wang / img.fact
    # lines_wang = lines_wang.astype(int)

    i = 0
    line_wang_polar = np.empty((lines_wang.shape[0], 1, 2))
    for line in lines_wang:
        for x1,y1,x2,y2 in line:
            line_wang_polar[i] = (radius(x1,y1,x2,y2), theta(x1,y1,x2,y2))
            cv2.line(line_image_wang,(x1,y1),(x2,y2),(0,0,250), round(2/img.fact))
            i += 1

    hough_wang = cv2.addWeighted(gray3ch, 0.5, line_image_wang, 0.8, 0)

    # cv2.imwrite("test/{}2hough_wang.jpg".format(img.basename), hough_wang)

    hough_on_canny_wang = cv2.addWeighted(cv2.bitwise_not(gray3ch_canny_wang), 0.2, line_image_wang, 0.8, 0)

    # cv2.imwrite("test/{}3hough_on_canny_wang.jpg".format(img.basename), hough_on_canny_wang)

def find_thetas(img):
    img_wang = wang_filter(img.small)
    lines_wang = find_straight_lines(img)
    draw_hough(lines_wang)

    # index = lines_wang[:,0,0].argmax()
    # lin = lines_wang[index, 0, :]

    # fig_wang = plt.figure()
    # ax = fig_wang.add_subplot(111, xlabel='angle', ylabel='radius', xlim=(-90, 90))
    # ax.plot(line_wang_polar[:,0,1], line_wang_polar[:,0,0],
    #         linestyle='', marker='.', color='blue', label='wang', alpha=0.8)
    # ax.legend()
    # fig_wang.savefig('test/{}4_polar.png'.format(img.basename))

    return np.array([25, 55])

def wang_filter(img):
    f = np.copy(img/255)
    W = np.copy(img/255) * 0
    N = np.copy(img/255) * 0
    g1 = np.copy(img/255)
    g2 = np.copy(g1)
    g3 = np.copy(g2)

    libwang = ct.CDLL("./libwang.so")
    libwang_wang_filter = libwang.wang_filter

    libwang_wang_filter.restype = None
    libwang_wang_filter.argtypes = [ndp(ct.c_double, flags="C_CONTIGUOUS"),
                                    ct.c_size_t, ct.c_size_t,
                                    ndp(ct.c_double, flags="C_CONTIGUOUS"),
                                    ndp(ct.c_double, flags="C_CONTIGUOUS"),
                                    ndp(ct.c_double, flags="C_CONTIGUOUS")]

    libwang_wang_filter(f, f.shape[0], f.shape[1], W, N, g1)
    libwang_wang_filter(g1, f.shape[0], f.shape[1], W, N, g2)
    libwang_wang_filter(g2, f.shape[0], f.shape[1], W, N, g3)

    G = np.array(g3*255, dtype='uint8')
    return G

def find_board(img):
    img.thetas = find_thetas(img)
    img_high = high_theta(img)
    img_wang = wang_filter(img_high)
    find_straight_lines(img_wang)

    # ret, cvthr = cv2.threshold(img.gray, 160, 255, cv2.THRESH_BINARY)
    return (10, 300, 110, 310)
