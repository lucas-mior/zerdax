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

def closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False):
    a0 = np.array(a0)
    a1 = np.array(a1)
    b0 = np.array(b0)
    b1 = np.array(b1)

    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
    '''

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0=True
        clampA1=True
        clampB0=True
        clampB1=True


    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
    
    _A = A / magA
    _B = B / magB
    
    cross = np.cross(_A, _B);
    denom = np.linalg.norm(cross)**2
    
    
    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A,(b0-a0))
        
        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A,(b1-a0))
            
            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0,b0,np.linalg.norm(a0-b0)
                    return a0,b1,np.linalg.norm(a0-b1)
                
                
            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1,b0,np.linalg.norm(a1-b0)
                    return a1,b1,np.linalg.norm(a1-b1)
                
                
        # Segments overlap, return distance between parallel segments
        return None,None,np.linalg.norm(((d0*_A)+a0)-b0)
        
    
    
    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0);
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA/denom;
    t1 = detB/denom;

    pA = a0 + (_A * t0) # Projected closest point on segment A
    pB = b0 + (_B * t1) # Projected closest point on segment B


    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1
        
        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1
            
        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B,(pA-b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)
    
        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A,(pB-a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    
    return pA,pB,np.linalg.norm(pA-pB)

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
    print("variance: ", var)
    tol_wrap = np.clip(var/8 + 35, 40, 50)
    tol_err  = np.clip(var/8,      15, 25)
    print("tol_wrap: ", tol_wrap)
    print("tol_err: ", tol_err)

    for a in A[:, 5]:
        err = abs(a - mean)
        if err > tol_wrap:
            rem[i] = 0
            C[0,:] = np.copy(A[i,:])
            C[0,5] = -C[0,5]
            B = np.append(B, C, axis=0)
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
        intersections, newlines = find_intersections(B, A)
    else:
        # B is more vertical, A is more horizontal
        A = A[A[:, 0, 1].argsort()]
        B = B[B[:, 0, 0].argsort()]
        intersections, newlines = find_intersections(A, B)

    i = 0
    rem = np.empty(A.shape[0])
    rem = np.int32(rem)
    aux = np.zeros((A.shape[0], 1, 7))
    aux[:,0, 0:6] = np.copy(A[:,0, 0:6])
    A = np.copy(aux)
    np.set_printoptions(linewidth=160)

    for a in A[:,0,:]:
        total = 0
        for b in B[:,0,:]:
            cl = closestDistanceBetweenLines([a[0],a[1],0],[a[2],a[3],0],[b[0],b[1],0],[b[2],b[3],0])
            total += cl[2]
        print("total: ", total)
        A[i,0,6] = total
        i += 1
    print("A: ", A)
    exit()

    join = np.concatenate((A,B))
    # join = np.empty((newlines.shape[0], 1, 4), dtype='int32')
    # join[:,0,0:4] = newlines
    draw_hough(img, join[:,:,0:4], img.small, c_thl, c_thh, h_th, h_minl, h_maxg, 1)

    circles = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR) * 0
    gray3ch = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR)

    for p in intersections:
        cv2.circle(circles, p, radius=6, color=(255, 0, 0), thickness=-1)

    image = cv2.addWeighted(gray3ch, 0.5, circles, 0.8, 0)
    cv2.imwrite("1{}4_circle.jpg".format(img.basename), image)

    return (10, 300, 110, 310)
