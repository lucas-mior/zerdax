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
    newA = []
    newB = []
    alin = np.zeros((A.shape[0], 5))
    blin = np.zeros((B.shape[0], 5))

    i = 0
    k = 0
    for r in A[:,0]:
        l1 = [(r[0],r[1]), (r[2],r[3])]
        k = 0
        for s in B[:,0]:
            l2 = [(s[0],s[1]), (s[2],s[3])]

            xdiff = (l1[0][0] - l1[1][0], l2[0][0] - l2[1][0])
            ydiff = (l1[0][1] - l1[1][1], l2[0][1] - l2[1][1])

            div = det(xdiff, ydiff)
            if div == 0:
                k += 1
                continue

            d = (det(*l1), det(*l2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div

            if x > 1000 or y > 562 or x <= 0 or y <= 0:
                k += 1
                continue
            else:
                inter.append((x,y))

                if blin[k,0] == 0:
                    blin[k,1] = x
                    blin[k,2] = y
                blin[k,0] += 1
                ran = radius(blin[k,1],blin[k,2],blin[k,3],blin[k,4])
                rno = radius(blin[k,1],blin[k,2],x,y)
                if rno - ran > -1:
                    blin[k,3] = x
                    blin[k,4] = y
                k += 1

                if alin[i,0] == 0:
                    alin[i,1] = x
                    alin[i,2] = y
                alin[i,0] += 1
                ran = radius(alin[i,1],alin[i,2],alin[i,3],alin[i,4])
                rno = radius(alin[i,1],alin[i,2],x,y)
                if rno - ran > -0.5:
                    alin[i,3] = x
                    alin[i,4] = y
        i += 1

    for i in range(0, len(alin)):
        if alin[i,0] > 10:
            newA.append([alin[i,1],alin[i,2],alin[i,3],alin[i,4]])

    for i in range(0, len(blin)):
        if blin[i,0] > 10:
            newB.append([blin[i,1],blin[i,2],blin[i,3],blin[i,4]])

    newA = np.array(newA, dtype='int32')
    newB = np.array(newB, dtype='int32')
    inter = np.array(inter, dtype='int32') 

    return inter, newA, newB

def radius(x1,y1,x2,y2):
    if x2 == 0 or y2 == 0 or x1 == 0 or y1 == 0:
        return 0
    else:
        return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def theta(x1,y1,x2,y2):
    print("calculating:", x1, y1, x2, y2)
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
    tol_wrap = np.clip(var/8 + 35, 40, 50)
    tol_err  = np.clip(var/8,      15, 25)

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

    lines = cv2.HoughLinesP(img_canny, 2, np.pi / 180,  h_th,  None, h_minl,    h_maxg)
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

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
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
    fig.savefig('1{}3_kmeans0.png'.format(img.basename))

    centers[0], A, B = remove_outliers(A, B, centers[0])
    centers[1], B, A = remove_outliers(B, A, centers[1])
    centers[0], A, B = remove_outliers(A, B, centers[0])
    centers[1], B, A = remove_outliers(B, A, centers[1])

    fig = plt.figure()
    plt.hist(A[:,5], 180, [-90, 90], color = 'red')
    plt.hist(B[:,5], 180, [-90, 90], color = 'blue')
    plt.hist(centers, 45, [-90, 90], color = 'yellow')
    fig.savefig('1{}3_kmeans1.png'.format(img.basename))

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

    auxA = np.empty((A.shape[0], 1, A.shape[1]))
    auxB = np.empty((B.shape[0], 1, B.shape[1]))
    auxA[:,0,:] = A
    auxB[:,0,:] = B
    A = np.array(auxA, dtype='int32')
    B = np.array(auxB, dtype='int32')

    if abs(img.thetas[0]) > abs(img.thetas[1]):
        # A is more vertical, B is more horizontal
        A = A[A[:, 0, 0].argsort()]
        B = B[B[:, 0, 1].argsort()]
        intersections, A, B = find_intersections(A, B)
    else:
        # B is more vertical, A is more horizontal
        A = A[A[:, 0, 1].argsort()]
        B = B[B[:, 0, 0].argsort()]
        intersections, A, B = find_intersections(B, A)

    # newA = np.empty((A.shape[0], 1, 4), dtype='int32')
    # newB = np.empty((B.shape[0], 1, 4), dtype='int32')
    # newA[:,0,:] = A
    # newB[:,0,:] = B
    # A = newA
    # B = newB
    # join = np.concatenate((A,B))

    # draw_hough(img, A,    img.small, c_thl, c_thh, h_th, h_minl, h_maxg, "A")
    # draw_hough(img, B,    img.small, c_thl, c_thh, h_th, h_minl, h_maxg, "B")
    # draw_hough(img, join, img.small, c_thl, c_thh, h_th, h_minl, h_maxg, "join")

    circles = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR) * 0
    gray3ch = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR)

    for p in intersections:
        cv2.circle(circles, p, radius=6, color=(255, 0, 0), thickness=-1)

    image = cv2.addWeighted(gray3ch, 0.5, circles, 0.8, 0)
    cv2.imwrite("1{}4_circl_{}_{}_{}_{}_{}.jpg".format(img.basename, c_thl, c_thh, h_th, h_minl, h_maxg), image)

    line_image = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR) * 0

    for x1, y1 in intersections:    
        distance = 0
        secondx = []
        secondy = []
        dist_listappend = []
        sort = []   
        a1 = False
        a2 = False
        for x2, y2 in intersections:      
            if (x1, y1) == (x2, y2):
                pass     
            else:
                aux = abs(abs(x1 - x2) - abs(y1-y2))
                distance = radius(x1,y1,x2,y2)
                angle = theta(x1,y1,x2,y2)
                if distance > 10 and aux < 50:
                    if abs(t - img.thetas[0]) > 10
                        a1 = True 
                        secondx.append(x2)
                        secondy.append(y2)
                        dist_listappend.append(distance)               
                    elif abs(t - img.thetas[1]) > 10: 
                        a2 = True
                        secondx.append(x2)
                        secondy.append(y2)
                        dist_listappend.append(distance)               
                elif aux < 100 and aux > 50:
                    if a1 and a2:
                        secondx.append(x2)
                        secondy.append(y2)
                        dist_listappend.append(distance)               
                else:
                    continue
        secondxy = list(zip(dist_listappend,secondx,secondy))
        secondxy = np.array(secondxy)
        sort = secondxy[secondxy[:,0].argsort()]
        medd = np.median(sort[0:3, 0])
        for con in range(0, 6):
            neg = (sort[con,1], sort[con,2])
            if sort[con,0] - medd > 20:
                continue
            cv2.line(line_image, (x1,y1), (int(neg[0]), int(neg[1])), (0,0,255), round(2/img.fact))

    conn = cv2.addWeighted(gray3ch, 0.5, line_image, 0.8, 0)
    conn = cv2.addWeighted(conn, 0.5, circles, 0.8, 0)

    cv2.imwrite('connected.png', conn)

    return (10, 300, 110, 310)
