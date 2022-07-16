import cv2
import numpy as np
import math
import sys
from Image import Image
from angles import set_kernels
from pathlib import Path
from aux import save,savefig

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import lwang

def lines_radius_theta(lines):
    aux = np.zeros((lines.shape[0], 1, 6), dtype='int32')
    aux[:,0,0:4] = np.copy(lines[:,0,0:4])
    lines = aux
    i = 0
    for line in lines:
        for x1,y1,x2,y2,r,t in line:
            lines[i, 0, 4] = round(radius(x1,y1,x2,y2))
            lines[i, 0, 5] = round(theta(x1,y1,x2,y2))
            i += 1
    return lines

def determinant(a, b):
    return a[0]*b[1] - a[1]*b[0]

def find_intersections(img, lines):
    inter = []
    last = (0,0)

    i = 0
    for x1,y1,x2,y2,r,t in lines:
        l1 = [(x1,y1), (x2,y2)]
        j = 0
        for xx1,yy1,xx2,yy2,rr,tt in lines:
            l2 =  [(xx1,yy1), (xx2,yy2)]
            if (x1,y1) == (xx1,yy1) and (x2,y2) == (xx2,yy2):
                continue

            if abs(t - tt) < 30:
                continue

            xdiff = (l1[0][0] - l1[1][0], l2[0][0] - l2[1][0])
            ydiff = (l1[0][1] - l1[1][1], l2[0][1] - l2[1][1])

            div = determinant(xdiff, ydiff)
            if div == 0:
                j += 1
                continue

            d = (determinant(*l1), determinant(*l2))
            x = determinant(d, xdiff) / div
            y = determinant(d, ydiff) / div

            if x > img.swidth or y > img.sheigth or x < 0 or y < 0:
                j += 1
                continue
            else:
                j += 1
                if radius(last[0], last[1], x, y) > 10:
                    inter.append((x,y))
                    last = (x,y)
                else:
                    print("Close point ignored: ({},{}) ~ ({},{})".format(last[0],last[1],x,y))
                    continue
        i += 1

    inter = np.array(inter, dtype='int32')
    return inter

def radius(x1,y1,x2,y2):
    return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def theta(x1,y1,x2,y2):
    return math.degrees(math.atan2((y1-y2),(x2-x1)))

def find_best_cont(img, img_wang, amin):
    got = False
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    kd = 5
    lasta = 0
    while kd <= 30:
        k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (kd,kd))
        dilate = cv2.morphologyEx(img_wang, cv2.MORPH_DILATE, k_dil)
        edges_gray = cv2.divide(img_wang, dilate, scale = 255)
        edges_bin = cv2.bitwise_not(cv2.threshold(edges_gray, 0, 255, cv2.THRESH_OTSU)[1])
        edges_opened = cv2.morphologyEx(edges_bin, cv2.MORPH_OPEN, ko, iterations = 1)
        contours, _ = cv2.findContours(edges_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in contours]
        perim = [cv2.arcLength(c, True) for c in contours]
        max_index = np.argmax(areas)
        a = areas[max_index]
        if a > amin:
            print("{} > {}, p = {}".format(a, amin, perim[max_index]))
            got = True
            break
        elif a > lasta - 20000:
            print("{} < {}, p = {}".format(a, amin, perim[max_index]))
            kd += 1
            lasta = a
        else:
            print("{} < {}: failed. p = {}".format(a, amin, perim[max_index]))
            break

    return contours, max_index

def find_hull(img):
    img_wang = lwang.wang_filter(img.sgray)

    contours,max_index = find_best_cont(img, img_wang, 0.25*img.sarea)

    img_contour = np.empty(img.gray3ch.shape, dtype='uint8') * 0
    cont = contours[max_index]
    hull = cv2.convexHull(cont)
    cv2.drawContours(img_contour, [hull], -1, (0, 255, 0), thickness=3)
    cv2.drawContours(img_contour, cont,   -1, (255,0,0), thickness=3)
    img_contour_drawn = cv2.addWeighted(img.gray3ch, 0.5, img_contour, 0.8, 0)

    return hull

def broad_hull(img, hull):
    Pxmin = hull[np.argmin(hull[:,0,0]),0]
    Pxmax = hull[np.argmax(hull[:,0,0]),0]
    Pymin = hull[np.argmin(hull[:,0,1]),0]
    Pymax = hull[np.argmax(hull[:,0,1]),0]

    Pxmin[0] = max(0, Pxmin[0]-20)
    Pymin[1] = max(0, Pymin[1]-70) # peças de trás vao além
    Pxmax[0] = min(img.swidth, Pxmax[0]+20)
    Pymax[1] = min(img.sheigth, Pymax[1]+20)

    if Pxmin[1] < Pxmax[1]:
        Pxmin[1] -= 20
        Pxmax[1] += 20
    else:
        Pxmax[1] -= 20
        Pxmin[1] += 20

    if Pymin[0] < Pymax[0]:
        Pymin[0] -= 20
        Pymax[0] += 20
    else:
        Pymax[0] -= 20
        Pymin[0] += 20

    return [Pymin[1],Pymax[1]], [Pxmin[0],Pxmax[0]]

def reduce_hull(img):
    img.hwidth = 900
    img.hfact = img.hwidth / img.hull.shape[1]
    img.hheigth = round(img.hfact * img.hull.shape[0])

    img.hull = cv2.resize(img.hull, (img.hwidth, img.hheigth))
    img.harea = img.hwidth * img.hheigth
    return img

def find_board(img):
    hull = find_hull(img)
    limx, limy = broad_hull(img, hull)

    limx[0] = round(limx[0] / img.sfact)
    limx[1] = round(limx[1] / img.sfact)
    limy[0] = round(limy[0] / img.sfact)
    limy[1] = round(limy[1] / img.sfact)

    img.hull = img.gray[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.hxoff = limx[0]
    img.hyoff = limy[0]
    img = reduce_hull(img)
    img.hull3ch = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR)
    img_wang = lwang.wang_filter(img.hull)

    lines, angles, c_thrl, c_thrh = try_impossible(img, img_wang)
    contours, max_index = magic_angle(img, angles, c_thrl, c_thrh)

    img_contour = np.empty(img.gray3ch.shape, dtype='uint8') * 0
    cont = contours[max_index]
    hull = cv2.convexHull(cont)
    cv2.drawContours(img_contour, [hull], -1, (0, 255, 0), thickness=3)
    cv2.drawContours(img_contour, cont,   -1, (255,0,0), thickness=3)
    img_contour_drawn = cv2.addWeighted(img.gray3ch, 0.5, img_contour, 0.8, 0)

    return (10, 300, 110, 310)

def try_impossible(img, img_wang):
    c_thrl0 = 100
    c_thrh0 = 200
    c_thrl = c_thrl0
    c_thrh = c_thrh0
    got_canny = True
    got_hough = False

    while c_thrl > 10 and c_thrh > 50:
        img_canny = cv2.Canny(img_wang, c_thrl, c_thrh)
        contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        a = areas[max_index]
        amin = 0.3 * img.harea
        if a > amin:
            print("{} > {}, @ {}, {}".format(a, amin, c_thrl, c_thrh))
            got_canny = True
            break
        else:
            if amin - a < amin:
                print("{} < {}, @ {}, {}".format(a, amin, c_thrl, c_thrh))
            c_thrl -= 9
            c_thrh -= 18

    h_maxg0 = 2
    h_minl0 = round((img.hwidth + img.hheigth)*0.1)
    h_thrv0 = round(h_minl0 / 6)
    h_angl0 = np.pi / 720

    if got_canny:
        h_maxg = h_maxg0
        h_minl = h_minl0
        h_thrv = h_thrv0
        h_angl = h_angl0
        while h_maxg < 10 and h_minl > (h_minl0 / 4) and h_angl < (np.pi / 180):
            lines = cv2.HoughLinesP(img_canny, 1, h_angl,  h_thrv,  None, h_minl, h_maxg)
            print("HOUGH @ {}, {}, {}, {}, {}".format(c_thrl, c_thrh, h_thrv, h_minl, h_maxg))
            if lines is not None and lines.shape[0] >= 4 + 10:
                lines = lines_radius_theta(lines)
                lines = filter_lines(img, lines)
                lines, angles = lines_kmeans(img, lines)
                print("angles: ", angles)
                inter = find_intersections(img, lines[:,0,:])
                if inter.shape[0] >= 20:
                    got_hough = True
                    break
            h_maxg += 1
            h_minl -= 10
            h_thrv = round(h_minl / 6)
            h_angl += np.pi/3600
    else:
        print("canny failed")

    if got_canny:
        save(img, "canny", img_canny)
        pass
    if got_hough:
        drawn_lines = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR) * 0
        for line in lines:
            for x1,y1,x2,y2,r,t in line:
                cv2.line(drawn_lines,(x1,y1),(x2,y2),(0,0,250),round(2/img.sfact))
        drawn_lines = cv2.addWeighted(img.hull3ch, 0.5, drawn_lines, 0.8, 0)
        save(img, "hough", drawn_lines)

        drawn_circles = np.copy(img.hull3ch) * 0
        for p in inter:
            cv2.circle(drawn_circles, p, radius=7, color=(255, 0, 0), thickness=-1)
        drawn_circles = cv2.addWeighted(img.hull3ch, 0.5, drawn_circles, 0.8, 0)
        save(img, "intersections".format(img.basename), drawn_circles)

    return lines, angles, c_thrl, c_thrh

def filter_lines(img, lines):
    rem = np.empty(lines.shape[0])
    rem = np.int32(rem)

    i = 0
    for line in lines:
        for x1,y1,x2,y2,r,t in line:
            if x1 < 3 and x2 < 3 or y1 < 3 and y2 < 3:
                rem[i] = 1
            elif (img.hwidth - x1 < 3) and (img.hwidth - x2) < 3 or (img.hheigth - y1) < 3 and (img.hheigth - y2) < 3:
                rem[i] = 1
            else:
                rem[i] = 0
        i += 1

    A = lines[rem==0]
    lines = A
    return lines

def lines_kmeans(img, lines):
    lines = np.float32(lines)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv2.kmeans(lines[:,:,5], 3, None, criteria, 10, flags)

    A = lines[labels==0]
    B = lines[labels==1]
    C = lines[labels==2]

    print("compactness: ", compactness)

    fig = plt.figure()
    plt.xticks(range(-90, 91, 10))
    plt.hist(A[:,5], 180, [-90, 90], color = (0.9, 0.0, 0.0, 0.9))
    plt.hist(B[:,5], 180, [-90, 90], color = (0.0, 0.0, 0.9, 0.9))
    plt.hist(C[:,5], 180, [-90, 90], color = (0.0, 0.9, 0.0, 0.9))
    plt.hist(centers, 20, [-90, 90], color = (0.7, 0.7, 0.0, 0.8))
    savefig(img, "kmeans0", fig)

    d1 = abs(centers[0] - centers[1])
    d2 = abs(centers[0] - centers[2])
    d3 = abs(centers[1] - centers[2])

    dd1 = d1 < 22.5 and d2 > 22.5 and d3 > 22.5
    dd2 = d2 < 22.5 and d1 > 22.5 and d3 > 22.5
    dd3 = d3 < 22.5 and d1 > 22.5 and d2 > 22.5

    if dd1 or dd2 or dd3:
        compactness,labels,centers = cv2.kmeans(lines[:,:,5], 2, None, criteria, 10, flags)
        A = lines[labels==0]
        B = lines[labels==1]
        fig = plt.figure()
        plt.xticks(range(-90, 90, 10))
        plt.hist(A[:,5], 180, [-90, 90], color = (0.9, 0.0, 0.0, 0.9))
        plt.hist(B[:,5], 180, [-90, 90], color = (0.0, 0.0, 0.9, 0.9))
        plt.hist(centers, 20, [-90, 90], color = (0.7, 0.7, 0.0, 0.7))
        savefig(img, "kmeans1", fig)

    lines = np.int32(lines)
    return lines, centers

def magic_angle(img, angles, c_thrl, c_thrh):
    img_wang = lwang.wang_filter(img.sgray)

    kernels = set_kernels(angles)
    i = 0
    k0 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    boost = np.zeros((kernels.shape[0], img_wang.shape[0], img_wang.shape[1]), dtype='uint8')
    for k in kernels:
        boost[i] = cv2.morphologyEx(img_wang, cv2.MORPH_DILATE, k, iterations = 1)
        save(img, "{}boost".format(i), boost[i])
        boost[i] = cv2.divide(img_wang, boost[i], scale = 255)
        save(img, "{}boost".format(i), boost[i])
        boost[i] = cv2.bitwise_not(cv2.threshold(boost[i], 0, 255, cv2.THRESH_OTSU)[1])
        save(img, "{}boost".format(i), boost[i])
        boost[i] = cv2.morphologyEx(boost[i], cv2.MORPH_OPEN, k, iterations = 1)
        save(img, "{}boost".format(i), boost[i])
        i += 1

    img_dil = boost.sum(axis=0, dtype='uint8')
    contours, _ = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cont = contours[max_index]
    hull = cv2.convexHull(cont)
    img_contour = np.empty(img.gray3ch.shape, dtype='uint8') * 0
    cv2.drawContours(img_contour, [hull], -1, (0, 255, 0), thickness=3)
    cv2.drawContours(img_contour, cont,   -1, (255,0,0), thickness=3)
    img_contour = cv2.addWeighted(img.gray3ch, 0.5, img_contour, 0.8, 0)

    save(img, "edges", img_dil)
    # save(img, "contor", img_contour)

    return contours, max_index
