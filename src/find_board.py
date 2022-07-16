import cv2
import numpy as np
import math
import sys
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from Image import Image
from aux import *
import lwang

def find_board(img):
    edges_opened, hull = find_hull(img)
    limx, limy = broad_hull(img, hull)

    img.edges = edges_opened[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    limx[0] = round(limx[0] / img.sfact)
    limx[1] = round(limx[1] / img.sfact)
    limy[0] = round(limy[0] / img.sfact)
    limy[1] = round(limy[1] / img.sfact)

    img.hull = img.gray[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.hxoff = limx[0]
    img.hyoff = limy[0]
    img = reduce_hull(img)
    img.hull3ch = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR)

    save(img, "edges", img.edges)
    img.angles, img.select_lines = find_angles(img)
    lines = try_impossible(img)

    return (10, 300, 110, 310)

def find_hull(img):
    img_wang = lwang.wang_filter(img.sgray)

    edges_opened,contours,max_index = find_best_cont(img, img_wang, 0.25*img.sarea)

    cont = contours[max_index]
    hull = cv2.convexHull(cont)
    # drawn_contours = np.empty(img.gray3ch.shape, dtype='uint8') * 0
    # cv2.drawContours(drawn_contours, [hull], -1, (0, 255, 0), thickness=3)
    # cv2.drawContours(drawn_contours, cont,   -1, (255,0,0), thickness=3)
    # drawn_contours = cv2.addWeighted(img.gray3ch, 0.5, drawn_contours, 0.8, 0)

    return edges_opened, hull

def find_best_cont(img, img_wang, amin):
    got = False
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    kd = 3
    while kd <= 50:
        k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (kd+round(kd/2),kd))
        print("kdil:", k_dil)
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
        else:
            print("{} < {}, p = {}".format(a, amin, perim[max_index]))
            kd += 1

    return edges_opened, contours, max_index

def broad_hull(img, hull):
    Pxmin = hull[np.argmin(hull[:,0,0]),0]
    Pxmax = hull[np.argmax(hull[:,0,0]),0]
    Pymin = hull[np.argmin(hull[:,0,1]),0]
    Pymax = hull[np.argmax(hull[:,0,1]),0]

    Pxmin[0] = max(0, Pxmin[0]-20)
    Pymin[1] = max(0, Pymin[1]-70) # peças de trás vao além
    Pxmax[0] = min(img.swidth, Pxmax[0]+20)
    Pymax[1] = min(img.sheigth, Pymax[1]+20)

    return [Pymin[1],Pymax[1]], [Pxmin[0],Pxmax[0]]

def find_angles(img):
    c_thrl0 = 100
    c_thrh0 = 200
    c_thrl = c_thrl0
    c_thrh = c_thrh0

    img_wang = lwang.wang_filter(img.hull)
    while c_thrl > 10 and c_thrh > 50:
        img_canny = cv2.Canny(img_wang, c_thrl, c_thrh)
        contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        a = areas[max_index]
        amin = 0.3 * img.harea
        if a > amin:
            print("{} > {}, @ {}, {}".format(a, amin, c_thrl, c_thrh))
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
            break
        h_maxg += 1
        h_minl -= 10
        h_thrv = round(h_minl / 6)
        h_angl += np.pi/3600

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

    return angles, lines

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

            if x > img.hwidth or y > img.hheigth or x < 0 or y < 0:
                j += 1
                continue
            else:
                j += 1
                if radius(last[0], last[1], x, y) > 10:
                    inter.append((x,y))
                    last = (x,y)
                else:
                    # print("Close point ignored: ({},{}) ~ ({},{})".format(last[0],last[1],x,y))
                    continue
        i += 1

    inter = np.array(inter, dtype='int32')
    return inter

def reduce_hull(img):
    img.hwidth = 900
    img.hfact = img.hwidth / img.hull.shape[1]
    img.hheigth = round(img.hfact * img.hull.shape[0])

    img.hull = cv2.resize(img.hull, (img.hwidth, img.hheigth))
    img.edges = cv2.resize(img.edges, (img.hwidth, img.hheigth))
    img.harea = img.hwidth * img.hheigth
    return img

def try_impossible(img):
    got_hough = False
    h_maxg0 = 0
    h_minl0 = round((img.hwidth + img.hheigth)*0.05)
    h_thrv0 = round(h_minl0 / 3)
    h_angl0 = np.pi / 1440

    h_maxg = h_maxg0
    h_minl = h_minl0
    h_thrv = h_thrv0
    h_angl = h_angl0
    while h_angl < (np.pi / 180):
        lines = cv2.HoughLinesP(img.edges, 1, h_angl, h_thrv,  None, h_minl, h_maxg)
        print("impossible @ {}, {}, {}, {}".format(180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
        if lines is not None and lines.shape[0] >= 70:
            lines = lines_radius_theta(lines)
            lines = filter_lines(img, lines)
            lines = filter_angles(img, lines)
            aux = np.copy(img.select_lines)
            lines = np.append(lines, aux, axis=0)
            inter = find_intersections(img, lines[:,0,:])
            got_hough = True
            break
        # while h_maxg < 5:
        #     h_maxg += 1
        while h_minl > h_minl0 / 3:
            h_minl -= 8
            h_thrv = round(h_minl / 3)
        h_angl += np.pi / 7200

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
    else:
        print("FAILED @ {}, {}, {}, {}".format(180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))

    return lines

def filter_lines(img, lines):
    rem = np.empty(lines.shape[0])
    rem = np.int32(rem)

    i = 0
    for line in lines:
        for x1,y1,x2,y2,r,t in line:
            if x1 < 3 and x2 < 3 or y1 < 3 and y2 < 3:
                rem[i] = 1
            elif (img.hwidth - x1) < 3 and (img.hwidth - x2) < 3 or (img.hheigth - y1) < 3 and (img.hheigth - y2) < 3:
                rem[i] = 1
            elif (x1 < 3 or (img.hwidth - x1) < 3) and (y2 < 3 or (img.hheigth - y2) < 3):
                rem[i] = 1
            elif (x2 < 3 or (img.hwidth - x2) < 3) and (y1 < 3 or (img.hheigth - y1) < 3):
                rem[i] = 1
            else:
                rem[i] = 0
        i += 1

    A = lines[rem==0]
    lines = A
    return lines

def filter_angles(img, lines):
    rem = np.empty(lines.shape[0])
    rem = np.int32(rem)

    i = 0
    for line in lines:
        for x1,y1,x2,y2,r,t in line:
            if abs(t - img.angles[0]) > 12 and abs(t - img.angles[1]) > 12:
                if img.angles.shape == 2:
                    rem[i] = 1
                elif abs(t - img.angles[2]) > 12:
                    rem[i] = 1
                else:
                    rem[i] = 0
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

    # fig = plt.figure()
    # plt.xticks(range(-90, 91, 10))
    # plt.hist(A[:,5], 180, [-90, 90], color = (0.9, 0.0, 0.0, 0.9))
    # plt.hist(B[:,5], 180, [-90, 90], color = (0.0, 0.0, 0.9, 0.9))
    # plt.hist(C[:,5], 180, [-90, 90], color = (0.0, 0.9, 0.0, 0.9))
    # plt.hist(centers, 20, [-90, 90], color = (0.7, 0.7, 0.0, 0.8))
    # savefig(img, "kmeans0", fig)

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
        # fig = plt.figure()
        # plt.xticks(range(-90, 90, 10))
        # plt.hist(A[:,5], 180, [-90, 90], color = (0.9, 0.0, 0.0, 0.9))
        # plt.hist(B[:,5], 180, [-90, 90], color = (0.0, 0.0, 0.9, 0.9))
        # plt.hist(centers, 20, [-90, 90], color = (0.7, 0.7, 0.0, 0.7))
        # savefig(img, "kmeans1", fig)

    lines = np.int32(lines)
    return lines, centers

