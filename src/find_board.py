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

def find_straight_lines(basename, img_canny, h_th, h_minl, h_maxg):

    lines = cv2.HoughLinesP(img_canny, 2, np.pi / 180,  h_th,  None, h_minl,    h_maxg)
              # HoughLinesP(image,   RHo, theta,   threshold, lines, minLength, maxGap)
    # dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
    # Rho : The resolution of the parameter r in pixels. We use 1 pixel.
    # theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
    # threshold: The minimum number of intersections to "*detect*" a line
    # lines: A vector that will store the parameters (xstart,ystart,xend,yend) of the detected lines
    # minLineLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
    # maxLineGap: The maximum gap between two points to be considered in the same line.

    return lines

def draw_hough(basename, lines, img, img_canny, a, b, c, d, e):
    line_image = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR) * 0
    gray3ch = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR)

    gray3ch_canny = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,250), round(2/img.fact))

    hough = cv2.addWeighted(gray3ch, 0.5, line_image, 0.8, 0)

    cv2.imwrite("1{}2hough_{}_{}_{}_{}_{}.jpg".format(basename, a, b, c, d, e), hough)

def find_thetas(img, c_thl, c_thh, h_th, h_minl, h_maxg):
    img_wang = wang_filter(img.small)

    img_canny = cv2.Canny(img_wang, c_thl, c_thh, None, 3, True)

    cv2.imwrite("1{}1canny_{}_{}.jpg".format(img.basename, c_thl, c_thh), img_canny)

    lines = find_straight_lines(img.basename, img_canny, h_th, h_minl, h_maxg)
    print("lines: ", lines)
    draw_hough(img.basename, lines, img, img.small, c_thl, c_thh, h_th, h_minl, h_maxg)

    new = np.zeros((lines.shape[0], 1, 6))
    new[:,0,0:4] = np.copy(lines[:,0,0:4])
    lines = np.float32(new)
    i = 0
    for line in lines:
        for x1,y1,x2,y2,r,t in line:
            lines[i, 0, 4] = radius(x1,y1,x2,y2)
            lines[i, 0, 5] = theta(x1,y1,x2,y2)
            i += 1

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # Apply KMeans
    compactness,labels,centers = cv2.kmeans(lines[:,:,5], 2, None, criteria, 10, flags)

    if abs(centers[0] - centers[1]) < 30
        print("K-means failed. Exiting")
        exit()

    A = lines[labels==0]
    B = lines[labels==1]

    print("OLD centers: ", centers)

    fig = plt.figure()
    plt.hist(A[:,5], 180, [-90, 90], color = 'r')
    plt.hist(B[:,5], 180, [-90, 90], color = 'b')
    plt.hist(centers, 45, [-90, 90], color = 'y')
    fig.savefig('1{}4_kmeans0.png'.format(img.basename))

    remA = np.empty(A.shape[0])
    remA = np.int32(remA)

    i = 0
    for a in A[:, 5]:
        if abs(a - centers[0]) > 15:
            remA[i] = 0
        else:
            remA[i] = 1
        i += 1

    A = A[remA==1]

    remB = np.empty(B.shape[0])
    remB = np.int32(remB)

    i = 0
    for b in B[:, 5]:
        if abs(b - centers[1]) > 15:
            remB[i] = 0
        else:
            remB[i] = 1
        i += 1

    B = B[remB==1]

    centers[0] = np.mean(A[:,5])
    centers[1] = np.mean(B[:,5])
    print("NEW centers: ", centers)

    fig = plt.figure()
    plt.hist(A[:,5], 180, [-90, 90], color = 'r')
    plt.hist(B[:,5], 180, [-90, 90], color = 'b')
    plt.hist(centers, 45, [-90, 90], color = 'y')
    fig.savefig('1{}4_kmeans1.png'.format(img.basename))

    return np.array(centers), A, B

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

def find_board(img, c_thl, c_thh, h_th, h_minl, h_maxg):

    img.thetas, A, B = find_thetas(img, c_thl, c_thh, h_th, h_minl, h_maxg)
    print("thetas: ", img.thetas[:,0])
    print("shape:", A.shape)
    newA = np.empty((A.shape[0], 1, A.shape[1]))
    newB = np.empty((B.shape[0], 1, B.shape[1]))
    newA[:,0,:] = A
    newB[:,0,:] = B
    newA = np.array(newA, dtype='int32')
    newB = np.array(newB, dtype='int32')

    join = np.concatenate((newA,newB))
    print("join: ", join)
    draw_hough(img.basename, join[:,:,0:4], img, img.small, c_thl, c_thh, h_th, h_minl, h_maxg)

    return (10, 300, 110, 310)
