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

def det(a, b):
    return a[0]*b[1] - a[1]*b[0]

def find_intersections(A, B):
    """ finds intersections between more vertical (A)
        and more horizontal (B) infinite lines """
    inter = []
    newlines = []
    for r in A[:,0]:
        j = 0
        l1 = [(r[0],r[1]), (r[2],r[3])]
        for s in B[:,0]:

            # if len(inter) < 10:
            l2 = [(s[0],s[1]), (s[2],s[3])]

            xdiff = (l1[0][0] - l1[1][0], l2[0][0] - l2[1][0])
            ydiff = (l1[0][1] - l1[1][1], l2[0][1] - l2[1][1])

            div = det(xdiff, ydiff)
            if div == 0:
                continue

            d = (det(*l1), det(*l2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            if x > 1000 or y > 562 or x <= 0 or y <= 0:
                continue

            inter.append((x,y))
            j += 1
        if j > 6:
            newlines.append([l1[0][0], l1[0][1], x, y])

    newlines = np.array(newlines, dtype='int32')
    return np.array(inter, dtype='int32'), newlines

def radius(x1,y1,x2,y2):
    return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def theta(x1,y1,x2,y2):
    return math.degrees(math.atan2((y2-y1),(x2-x1)))

def draw_hough(basename, lines, img, img_canny, a, b, c, d, e, clean):
    line_image = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR) * 0
    gray3ch = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR)

    gray3ch_canny = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,250), round(2/img.fact))

    hough = cv2.addWeighted(gray3ch, 0.5, line_image, 0.8, 0)

    if clean == 1:
        cv2.imwrite("1{}2hough_{}_{}_{}_{}_{}_{}.jpg".format(basename, a, b, c, d, e, clean), hough)

def find_thetas(img, c_thl, c_thh, h_th, h_minl, h_maxg):
    img_wang = wang_filter(img.small)

    img_canny = cv2.Canny(img_wang, c_thl, c_thh, None, 3, True)

    # cv2.imwrite("1{}1canny_{}_{}.jpg".format(img.basename, c_thl, c_thh), img_canny)

    lines = cv2.HoughLinesP(img_canny, 2, np.pi / 180,  h_th,  None, h_minl,    h_maxg)

    draw_hough(img.basename, lines, img, img.small, c_thl, c_thh, h_th, h_minl, h_maxg, 0)

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

    if abs(centers[0] - centers[1]) < 30:
        print("K-means failed. Exiting")
        exit()

    A = lines[labels==0]
    B = lines[labels==1]

    fig = plt.figure()
    plt.hist(A[:,5], 180, [-90, 90], color = 'red')
    plt.hist(B[:,5], 180, [-90, 90], color = 'blue')
    plt.hist(centers, 45, [-90, 90], color = 'yellow')
    # fig.savefig('1{}4_kmeans0.png'.format(img.basename))

    remA = np.empty(A.shape[0])
    remA = np.int32(remA)

    corrected = False
    i = 0
    for a in A[:, 5]:
        if abs(a - centers[0]) > 15:
            remA[i] = 0
            corrected = True
        else:
            remA[i] = 1
        i += 1

    remB = np.empty(B.shape[0])
    remB = np.int32(remB)

    i = 0
    for b in B[:, 5]:
        if abs(b - centers[1]) > 15:
            remB[i] = 0
            corrected = True
        else:
            remB[i] = 1
        i += 1

    if corrected:
        A = A[remA==1]
        B = B[remB==1]
    else:
        return np.array(centers), A, B

    centers[0] = np.mean(A[:,5])
    centers[1] = np.mean(B[:,5])

    fig = plt.figure()
    plt.hist(A[:,5], 180, [-90, 90], color = 'r')
    plt.hist(B[:,5], 180, [-90, 90], color = 'b')
    plt.hist(centers, 45, [-90, 90], color = 'y')
    # fig.savefig('1{}4_kmeans1.png'.format(img.basename))

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

    newA = np.empty((A.shape[0], 1, A.shape[1]))
    newB = np.empty((B.shape[0], 1, B.shape[1]))
    newA[:,0,:] = A
    newB[:,0,:] = B
    newA = np.array(newA, dtype='int32')
    newB = np.array(newB, dtype='int32')

    if abs(img.thetas[0]) > abs(img.thetas[1]):
        A = newA[newA[:, 0, 0].argsort()]
        B = newB[newB[:, 0, 1].argsort()]
        intersections, newlines = find_intersections(B, A)
    else:
        A = newA[newA[:, 0, 1].argsort()]
        B = newB[newB[:, 0, 0].argsort()]
        intersections, newlines = find_intersections(A, B)

    # join = np.concatenate((A,B))
    join = np.empty((newlines.shape[0], 1, 4), dtype='int32')
    join[:,0,0:4] = newlines
    draw_hough(img.basename, join[:,:,0:4], img, img.small, c_thl, c_thh, h_th, h_minl, h_maxg, 1)

    circles = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR) * 0
    gray3ch = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR)

    for p in intersections:
        cv2.circle(circles, p, radius=6, color=(255, 0, 0), thickness=-1)

    image = cv2.addWeighted(gray3ch, 0.5, circles, 0.8, 0)
    cv2.imwrite("1{}5circle.jpg".format(img.basename), image)

    return (10, 300, 110, 310)
