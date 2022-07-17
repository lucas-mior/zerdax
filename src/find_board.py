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
    save(img, "sgray", img.sgray)
    img.wang = lwang.wang_filter(img.sgray)
    img.canny = find_canny(img)
    img.medges, img.hullxy = find_morph(img)
    limx, limy = broad_hull(img)

    img.medges = img.medges[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.wang = img.wang[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    limx[0] = round(limx[0] / img.sfact)
    limx[1] = round(limx[1] / img.sfact)
    limy[0] = round(limy[0] / img.sfact)
    limy[1] = round(limy[1] / img.sfact)

    img.hull = img.gray[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.hxoff = limx[0]
    img.hyoff = limy[0]
    img = reduce_hull(img)
    save(img, "hull", img.hull)
    img.hull3ch = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR)
    save(img, "edges", img.medges)

    img.canny = find_canny(img)
    img.angles, img.select_lines = find_angles(img)
    # lines = try_impossible(img)

    corners = (10, 300, 110, 310)
    return corners

def find_morph(img):
    medges,hullxy = find_best_cont(img)

    drawn_contours = np.empty(img.gray3ch.shape, dtype='uint8') * 0
    cv2.drawContours(drawn_contours, [hullxy], -1, (0, 255, 0), thickness=3)
    drawn_contours = cv2.addWeighted(img.gray3ch, 0.5, drawn_contours, 0.8, 0)
    save(img, "convex", drawn_contours)

    return medges, hullxy

def find_best_cont(img):
    Aok = 0.4 * img.sarea
    Ami = 0.3 * img.sarea
    alast = 0
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    kd = 3
    while kd <= 40:
        k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (kd+round(kd/2),kd))
        dilate = cv2.morphologyEx(img.wang, cv2.MORPH_DILATE, k_dil)
        edges_gray = cv2.divide(img.wang, dilate, scale = 255)
        edges_bin = cv2.bitwise_not(cv2.threshold(edges_gray, 0, 255, cv2.THRESH_OTSU)[1])
        edges_opened = cv2.morphologyEx(edges_bin, cv2.MORPH_OPEN, ko, iterations = 1)
        edges_opened += img.canny
        contours, _ = cv2.findContours(edges_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        hullxy = cv2.convexHull(contours[max_index])
        areas = [cv2.contourArea(hullxy)]
        max_index = np.argmax(areas)
        a = areas[max_index]
        if kd == 20:
            medges = edges_opened
        if a > Aok:
            print("{} > {} @ ksize = {}".format(a, Aok, kd))
            break
        elif kd > 30:
            print("{} < {} @ ksize = {}".format(a, Aok, kd))
            if a > alast - 10000:
                alast = a
                kd += 1
                continue
            elif a > aok:
                break
        else:
            print("{} < {} @ ksize = {}".format(a, Aok, kd))
            alast = a
            kd += 1
            pass

    if kd < 30:
        medges = edges_opened

    return medges,hullxy

def broad_hull(img):
    Pxmin = img.hullxy[np.argmin(img.hullxy[:,0,0]),0]
    Pxmax = img.hullxy[np.argmax(img.hullxy[:,0,0]),0]
    Pymin = img.hullxy[np.argmin(img.hullxy[:,0,1]),0]
    Pymax = img.hullxy[np.argmax(img.hullxy[:,0,1]),0]

    Pxmin[0] = max(0, Pxmin[0]-20)
    Pymin[1] = max(0, Pymin[1]-40) # peças de trás vao além
    Pxmax[0] = min(img.swidth, Pxmax[0]+20)
    Pymax[1] = min(img.sheigth, Pymax[1]+20)

    return [Pymin[1],Pymax[1]], [Pxmin[0],Pxmax[0]]

def find_canny(img):
    wmin = 6
    c_thrl0 = 80
    c_thrh0 = 200
    c_thrl = c_thrl0
    c_thrh = c_thrh0

    while c_thrh > 70:
        img.canny = cv2.Canny(img.wang, c_thrl, c_thrh)
        w = img.canny.mean()
        if w > wmin:
            print("{0:0=.2f} < {1}, @ {2}, {3}".format(w, wmin, c_thrl, c_thrh))
            break
        else:
            if wmin - w < wmin:
                print("{0:0=.2f} > {1}, @ {2}, {3}".format(w, wmin, c_thrl, c_thrh))
        if c_thrl > 10:
            c_thrl -= 9
        c_thrh -= 9

    save(img, "canny", img.canny)
    return img.canny

def find_angles(img):
    got_hough = False
    h_maxg0 = 2
    h_minl0 = round((img.hwidth + img.hheigth)*0.2)
    h_thrv0 = round(h_minl0 / 6)
    h_angl0 = np.pi / 360

    h_maxg = h_maxg0
    h_minl = h_minl0
    h_thrv = h_thrv0
    h_angl = h_angl0
    minlines = 24
    while h_angl < (np.pi / 30):
        lines = cv2.HoughLinesP(img.canny, 1, h_angl,  h_thrv,  None, h_minl, h_maxg)
        if lines is not None and lines.shape[0] >= minlines:
            print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
            lines = radius_theta(lines)
            lines = filter_lines(img, lines)
            lines, angles = lines_kmeans(img, lines)
            print("angles: ", angles)
            got_hough = True
            break
        elif lines is not None:
            print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
        if h_maxg < 20:
            h_maxg += 1
        if h_minl > h_minl0 / 2:
            h_minl -= 10
        if h_angl > (np.pi / 40):
            h_minl = 32
            h_maxg = 2
            minlines = 60

        h_thrv = round(h_minl / 10)
        h_angl += np.pi/14400

    if not got_hough:
        lines = radius_theta(lines)
        lines = filter_lines(img, lines)
        lines, angles = lines_kmeans(img, lines)
        print("angles: ", angles)

    drawn_lines = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR) * 0
    for line in lines:
        for x1,y1,x2,y2,r,t in line:
            cv2.line(drawn_lines,(x1,y1),(x2,y2),(0,0,250),round(2/img.sfact))
    drawn_lines = cv2.addWeighted(img.hull3ch, 0.5, drawn_lines, 0.8, 0)
    save(img, "hough", drawn_lines)

    return angles, lines

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
    img.medges = cv2.resize(img.medges, (img.hwidth, img.hheigth))
    img.wang = cv2.resize(img.wang, (img.hwidth, img.hheigth))
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
        lines = cv2.HoughLinesP(img.medges, 1, h_angl, h_thrv,  None, h_minl, h_maxg)
        if lines is not None and lines.shape[0] >= 70:
            print("{} lines @ {}º, {}, {}, {}".format(lines.shape[0],180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
            lines = radius_theta(lines)
            lines = filter_lines(img, lines)
            lines = filter_angles(img, lines)
            aux = np.copy(img.select_lines)
            lines = np.append(lines, aux, axis=0)
            inter = find_intersections(img, lines[:,0,:])
            got_hough = True
            break
        # if h_maxg < 5:
        #     h_maxg += 1
        if h_minl > h_minl0 / 3:
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
