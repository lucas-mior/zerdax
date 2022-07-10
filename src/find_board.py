import cv2
import numpy as np
import math
import sys
from Image import Image
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import ctypes as ct
from numpy.ctypeslib import ndpointer as ndp

def radius(x1,y1,x2,y2):
    return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def theta(x1,y1,x2,y2):
    return math.degrees(math.atan2((y2-y1),(x2-x1)))

def high_theta(image, thetas, basename):
    print("high_theta(theta = {})", thetas)
    k450 = np.array([
                    [ -3.0,  -2.0, -1.0, -1.0, +0.0,],
                    [ -2.0,  -1.0, -1.0, +0.0, +1.0,],
                    [ -1.0,  -1.0, +0.0, +1.0, +1.0,],
                    [ -1.0,  +0.0, +1.0, +1.0, +2.0,],
                    [ -0.0,  +1.0, +1.0, +2.0, +3.0,],
                    ])
    # k451 = -k450
    k250 = np.array([
                    [ -2.0,  -2.0, -1.0, -1.0, +0.0,],
                    [ -2.0,  -1.0, -1.0, +0.0, +1.0,],
                    [ -1.0,  -1.0, +0.0, +1.0, +1.0,],
                    [ -1.0,  +0.0, +1.0, +1.0, +1.0,],
                    [ -0.0,  +1.0, +1.0, +1.0, +2.0,],
                    ])
    # k251 = -k250

    k450 = k450/(np.sum(k450) if np.sum(k450) != 0 else 1)
    # k451 = k451/(np.sum(k451) if np.sum(k451) != 0 else 1)
    k250 = k250/(np.sum(k250) if np.sum(k250) != 0 else 1)
    # k251 = k251/(np.sum(k251) if np.sum(k251) != 0 else 1)

    img_450 = cv2.filter2D(image, -1, k450)
    # img_451 = cv2.filter2D(image, -1, k451)
    img_250 = cv2.filter2D(image, -1, k250)
    # img_251 = cv2.filter2D(image, -1, k251)

    cv2.imwrite("1{}450.jpg".format(basename), img_450)
    # cv2.imwrite("1{}451.jpg".format(basename), img_451)
    cv2.imwrite("1{}250.jpg".format(basename), img_250)
    # cv2.imwrite("1{}251.jpg".format(basename), img_251)

    img_high = np.copy(img_450 + img_250)
    # img_high = img_high.astype(int)
    return img_high

def find_straight_lines(basename, img_canny, h_th, h_minl, h_maxg):

    lines = cv2.HoughLinesP(img_canny, 1, np.pi / 180,  h_th,        None, h_minl,   h_maxg)
                   # HoughLinesP(image,    RHo,       theta, threshold, lines, minLength, maxGap)
    # dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
    # Rho : The resolution of the parameter r in pixels. We use 1 pixel.
    # theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
    # threshold: The minimum number of intersections to "*detect*" a line
    # lines: A vector that will store the parameters (xstart,ystart,xend,yend) of the detected lines
    # minLineLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
    # maxLineGap: The maximum gap between two points to be considered in the same line.

    return lines

def draw_hough(basename, lines, img, img_canny, a, b, c, d, e):
    print(a, b, c, d, e)
    line_image = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR) * 0
    gray3ch = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR)

    gray3ch_canny = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)

    # lines = lines / img.fact
    # lines = lines.astype(int)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,250), round(2/img.fact))

    hough = cv2.addWeighted(gray3ch, 0.5, line_image, 0.8, 0)

    cv2.imwrite("1{}2hough_{}_{}_{}_{}_{}.jpg".format(basename, a, b, c, d, e), hough)
    # hough_on_canny = cv2.addWeighted(cv2.bitwise_not(gray3ch_canny), 0.2, line_image, 0.8, 0)
    # cv2.imwrite("1{}3hough_on_canny.jpg".format(basename), hough_on_canny)

def find_thetas(img, c_thl, c_thh):
    img_wang = wang_filter(img.small)

    img_canny = cv2.Canny(img_wang, c_thl, c_thh, None, 3, True)

    # cv2.imwrite("1{}1canny.jpg".format(img.basename), img_canny)

    lines = find_straight_lines(img.basename, img_canny, 30, 100, 15)
    draw_hough(img.basename, lines, img, img_canny)

    # index = lines[:,0,0].argmax()
    # lin = lines[index, 0, :]
    i = 0
    line_polar = np.empty((lines.shape[0], 1, 2))
    for line in lines:
        for x1,y1,x2,y2 in line:
            line_polar[i] = (radius(x1,y1,x2,y2), theta(x1,y1,x2,y2))
            i += 1

    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='angle', ylabel='radius', xlim=(-90, 90))
    ax.plot(line_polar[:,0,1], line_polar[:,0,0],
            linestyle='', marker='.', color='blue', label='line', alpha=0.8)
    ax.legend()
    fig.savefig('1{}4_polar.png'.format(img.basename))

    return np.array([-35, 45])

def wang_filter(image):
    f = np.copy(image/255)
    W = np.copy(image/255) * 0
    N = np.copy(image/255) * 0
    g = np.copy(image/255)

    libwang = ct.CDLL("./libwang.so")
    libwang_wang_filter = libwang.wang_filter

    libwang_wang_filter.restype = None
    libwang_wang_filter.argtypes = [ndp(ct.c_double, flags="C_CONTIGUOUS"),
                                    ct.c_size_t, ct.c_size_t,
                                    ndp(ct.c_double, flags="C_CONTIGUOUS"),
                                    ndp(ct.c_double, flags="C_CONTIGUOUS"),
                                    ndp(ct.c_double, flags="C_CONTIGUOUS")]

    libwang_wang_filter(f, f.shape[0], f.shape[1], W, N, g)
    libwang_wang_filter(g, f.shape[0], f.shape[1], W, N, f)
    libwang_wang_filter(f, f.shape[0], f.shape[1], W, N, g)

    G = np.array(g*255, dtype='uint8')
    return G

def find_board(img, c_thl = 30, c_thh = 150, h_th = 50, h_minl = 150, h_maxg = 15):

    # img.thetas = find_thetas(img)
    # print("theas: ", img.thetas)
    # img_high = high_theta(img.small, img.thetas, img.basename)
    # cv2.imwrite("test/0{}hightheta.jpg".format(img.basename), img_high)

    img_wang = wang_filter(img.small)
    img_canny = cv2.Canny(img_wang, c_thl, c_thh, None, 3, True)
    cv2.imwrite("1{}1canny_{}_{}.jpg".format(img.basename, c_thl, c_thh), img_canny)

    lines = find_straight_lines(img.basename, img_canny, h_th, h_minl, h_maxg)
    draw_hough(img.basename, lines, img, img.small, c_thl, c_thh, h_th, h_minl, h_maxg)

    return (10, 300, 110, 310)

def reduce(img):
    new_width = 1000
    img.fact = new_width / img.gray.shape[1]
    new_height = round(img.fact * img.gray.shape[0])

    dsize = (new_width, new_height)
    img.small = cv2.resize(img.gray, dsize)
    return img

if __name__ == "__main__":
    filename = sys.argv[1]
    img = Image(filename)
    img.basename = Path(filename).stem
    img.color = cv2.imread(filename)
    img.gray = cv2.cvtColor(img.color, cv2.COLOR_BGR2GRAY)
    img = reduce(img)
    img.thetas = [22, 45]
    img_high = high_theta(img.small, img.thetas, img.basename)
    cv2.imwrite("0{}hightheta.jpg".format(img.basename), img_high)
