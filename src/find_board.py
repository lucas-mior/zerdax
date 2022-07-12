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

def shortest_connections(img, intersections):
    line_image = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR) * 0

    for x1, y1 in intersections:    
        distance = 0
        secondx = []
        secondy = []
        dist_list = []
        angle_list = []
        sort = []   
        a1 = False
        a2 = False
        for x2, y2 in intersections:      
            if (x1, y1) == (x2, y2):
                continue
            else:
                distance = radius(x1,y1,x2,y2)
                angle = theta(x1,y1,x2,y2)
                secondx.append(x2)
                secondy.append(y2)
                dist_list.append(distance)               
                angle_list.append(angle)               

        secondxy = list(zip(dist_list, angle_list, secondx, secondy))
        secondxy = np.array(secondxy)
        sort = secondxy[secondxy[:,0].argsort()]
        for con in range(0, len(sort)):
            neg = (sort[con,2], sort[con,3])
            if sort[con,0] < 50:
                cv2.line(line_image, (x1,y1), (round(neg[0]), round(neg[1])), (0,0,255), round(2/img.fact))
            else:
                continue

    return line_image

def det(a, b):
    return a[0]*b[1] - a[1]*b[0]

def find_intersections(lines):
    inter = []

    i = 0
    for r in lines:
        l1 = [(r[0],r[1]), (r[2],r[3])]
        j = 0
        for s in lines:
            if i == j:
                continue

            if abs(r[5] - s[5]) < 30:
                continue

            l2 = [(s[0],s[1]), (s[2],s[3])]

            xdiff = (l1[0][0] - l1[1][0], l2[0][0] - l2[1][0])
            ydiff = (l1[0][1] - l1[1][1], l2[0][1] - l2[1][1])

            div = det(xdiff, ydiff)
            if div == 0:
                j += 1
                continue

            d = (det(*l1), det(*l2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div

            if x > 1000 or y > 667 or x <= 0 or y <= 0:
                j += 1
                continue
            else:
                j += 1
                inter.append((x,y))
        i += 1

    inter = np.array(inter, dtype='int32')
    return inter

def radius(x1,y1,x2,y2):
    return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def theta(x1,y1,x2,y2):
    return math.degrees(math.atan2((y2-y1),(x2-x1)))

def draw_hough(img, lines, img_canny, c_thrl, c_thrh, h_thrv, h_minl, h_maxg, clean):
    line_image = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR) * 0
    gray3ch = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,250), round(2/img.fact))

    hough = cv2.addWeighted(gray3ch, 0.5, line_image, 0.8, 0)

    cv2.imwrite("1{}2_hough_{}_{}_{}_{}_{}_{}.jpg".format(img.basename, c_thrl, c_thrh, h_thrv, h_minl, h_maxg, clean), hough)

def remove_outliers(A, B, mean):
    rem = np.empty(A.shape[0])
    rem = np.int32(rem)

    corrected = False
    i = 0
    C = np.empty((1,6))

    var = np.var(A[:,5])
    # tol_wrap = np.clip(var/8 + 35, 40, 50)
    # tol_err  = np.clip(var/8,      15, 25)
    tol_wrap = 100000
    tol_err  = 100000

    for a in A[:, 5]:
        err = abs(a - mean)
        if err > tol_wrap:
            rem[i] = 0
            C[0,:] = np.copy(A[i,:])
            C[0,5] = -C[0,5]
            B = np.append(B, C, axis=0)
            corrected = True
        elif err > tol_err - 5:
            if abs(a) < 1 or abs(a) > 89:
                rem[i] = 0
                corrected = True
        elif err > tol_err:
            rem[i] = 0
            corrected = True
        else:
            rem[i] = 1
        i += 1

    if not corrected:
        return mean, A, B
    else:
        A = A[rem==1]
        mean = np.mean(A[:,5])
        return mean, A, B

def find_thetas(img, c_thl, c_thh, h_th, h_minl, h_maxg):
    img_wang = wang_filter(img.small)
    img_canny = cv2.Canny(img_wang, c_thl, c_thh, None, 3, True)
    cv2.imwrite("1{}1_canny_{}_{}.jpg".format(img.basename, c_thl, c_thh), img_canny)

    lines = cv2.HoughLinesP(img_canny, 2, np.pi / 180,  h_th,  None, h_minl, h_maxg)
    draw_hough(img, lines, img.small, c_thl, c_thh, h_th, h_minl, h_maxg, 0)

    aux = np.zeros((lines.shape[0], 1, 6))
    aux[:,0,0:4] = np.copy(lines[:,0,0:4])
    lines = np.float32(aux)
    i = 0
    for line in lines:
        for x1,y1,x2,y2,r,t in line:
            lines[i, 0, 4] = radius(x1,y1,x2,y2)
            lines[i, 0, 5] = theta(x1,y1,x2,y2)
            i += 1

    return lines

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

    lines = find_thetas(img, c_thl, c_thh, h_th, h_minl, h_maxg)

    intersections = find_intersections(lines[:,0,:])

    circles = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR) * 0
    gray3ch = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR)

    for p in intersections:
        cv2.circle(circles, p, radius=6, color=(255, 0, 0), thickness=-1)

    points = circles[:,:,0]
    cv2.imwrite("1{}4_points_{}_{}_{}_{}_{}.jpg".format(img.basename, c_thl, c_thh, h_th, h_minl, h_maxg), points)

    newlines = cv2.HoughLinesP(points, 1, np.pi / 180,  h_th, None, h_minl, 50)
    draw_hough(img, newlines, img.small, c_thl, c_thh, h_th, h_minl, h_maxg, "af")

    image = cv2.addWeighted(gray3ch, 0.5, circles, 0.8, 0)
    cv2.imwrite("1{}4_circl_{}_{}_{}_{}_{}.jpg".format(img.basename, c_thl, c_thh, h_th, h_minl, h_maxg), image)

    line_image = shortest_connections(img, intersections)
    conn = cv2.addWeighted(gray3ch, 0.5, line_image, 0.8, 0)
    cv2.imwrite('1{}5_conne_{}_{}_{}_{}_{}.jpg'.format(img.basename, c_thl, c_thh, h_th, h_minl, h_maxg), conn)

    return (10, 300, 110, 310)
