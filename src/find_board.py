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
from lines import HoughBundler
import lwang

def find_board(img):
    # save(img, "sgray", img.sgray)
    img.wang0 = lwang.wang_filter(img.sgray)
    # save(img, "wang0", img.wang0)
    img, a, apoly,cont = region(img)
    img.ext = False

    if img.poly.shape[0] <= 2 or (abs(apoly - a) > 0.04*img.sarea) or (not img.got_hull and abs(a) < (0.2 * img.sarea)):
        print("\033[31;1;1m========== CASO EXTREMO ===========\033[0;m")
        img.ext = True
        drawn_contours = np.empty(img.gray3ch.shape, dtype='uint8') * 0
        cv2.drawContours(drawn_contours, [img.hullxy], -1, (255, 255, 0), thickness=3)
        img.help = drawn_contours[:,:,0]
        img, a, apoly, _ = region(img, maxkd = 20, cmax = 20, nymax = 12, skip=True)

    drawn_contours = np.empty(img.gray3ch.shape, dtype='uint8') * 0
    cv2.drawContours(drawn_contours, cont, -1, (255, 0, 0), thickness=3)
    cv2.drawContours(drawn_contours, [img.poly], -1, (0, 0, 255), thickness=3)
    img.medges = cv2.bitwise_or(img.medges, drawn_contours[:,:,2])
    cv2.drawContours(drawn_contours, [img.hullxy], -1, (0, 255, 0), thickness=3)
    drawn_contours = cv2.addWeighted(img.gray3ch, 0.4, drawn_contours, 0.7, 0)
    save(img, "convex_poly", drawn_contours)

    # limx, limy = broad_hull(img)
    x,y,w,h = cv2.boundingRect(img.poly)
    limx = np.zeros((2), dtype='int32')
    limy = np.zeros((2), dtype='int32')
    limx[0] = max(y-15, 0)
    limx[1] = min(y+h+15, img.swidth)
    limy[0] = max(x-15, 0)
    limy[1] = max(x+w+15, img.sheigth)

    if not img.got_hull:
        print("finding board region failed [find_morph()]")

    img.medges = img.medges[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.wang = img.wang[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.clahe = img.clahe[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    limx[0] = round(limx[0] / img.sfact)
    limx[1] = round(limx[1] / img.sfact)
    limy[0] = round(limy[0] / img.sfact)
    limy[1] = round(limy[1] / img.sfact)

    img.hull = img.gray[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.hxoff = limx[0]
    img.hyoff = limy[0]
    img = reduce_hull(img)
    img.hull3ch = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR)

    # save(img, "hull", img.hull)

    img.canny = find_canny(img, wmin = 8)
    save(img, "canny_find_magic_angles", img.canny)
    img.medges += img.canny
    # save(img, "medges+canny", img.medges)
    img.angles, img.select_lines = find_angles(img)

    lines,inter = magic_lines(img)

    img.corners = find_corners(img, inter)

    return img

def find_morph(img, Amin, maxkd=12, skip=False):
    got_hull = False
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    kd = 5
    while kd <= maxkd:
        if kd > 12:
            kx = kd
        else:
            kx = kd+round(kd/3)
        k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (kd,kx))
        dilate = cv2.morphologyEx(img.wang, cv2.MORPH_DILATE, k_dil)
        edges_gray = cv2.divide(img.wang, dilate, scale = 255)
        edges_thr = cv2.threshold(edges_gray, 0, 255, cv2.THRESH_OTSU)[1]
        edges_bin = cv2.bitwise_not(edges_thr)

        edges_bin = cv2.morphologyEx(edges_bin, cv2.MORPH_ERODE, ko, iterations = 1)
        edges_wcanny = cv2.bitwise_or(img.canny, edges_bin)
        if skip:
            edges_wcanny = cv2.bitwise_or(img.help, edges_wcanny)
        contours, _ = cv2.findContours(edges_wcanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cont = contours[max_index]
        poly = cv2.approxPolyDP(cont,0.035*cv2.arcLength(cont,True),True)
        hullxy = cv2.convexHull(poly)
        a = cv2.contourArea(hullxy)
        apoly = cv2.contourArea(poly)
        if a > Amin:
            if poly.shape[0] < 4:
                kd += 1
                continue
            else:
                print("{} > {} @ ksize = {} [GOTHULL]".format(a, Amin, kd))
                got_hull = True
                break
        else:
            print("{} < {} @ ksize = {}".format(a, Amin, kd))
            kd += 1

    medges = edges_bin

    return medges, hullxy, got_hull, a, kd, poly, apoly, cont

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

def find_canny(img, wmin = 6):
    c_thrl0 = 80
    c_thrh0 = 200
    c_thrl = c_thrl0
    c_thrh = c_thrh0

    while c_thrh > 12:
        img.canny = cv2.Canny(img.wang, c_thrl, c_thrh)
        w = img.canny.mean()
        if w > wmin:
            print("{0:0=.2f} > {1}, @ {2}, {3}".format(w, wmin, c_thrl, c_thrh))
            break
        else:
            if wmin - w < wmin:
                print("{0:0=.2f} < {1}, @ {2}, {3}".format(w, wmin, c_thrl, c_thrh))
        c_thrl = max(2, c_thrl - 9)
        c_thrh -= 9

    if w < wmin:
        print("Canny failed: {0:0=.2f} < {1}, @ {2}, {3}".format(w, wmin, c_thrl, c_thrh))
        exit()

    return img.canny

def find_angles(img, getangles=True):
    got_hough = False
    h_maxg0 = 2
    if getangles:
        h_minl0 = round((img.hwidth + img.hheigth)*0.2)
    else:
        h_minl0 = round((img.swidth + img.sheigth)*0.2)
    h_thrv0 = round(h_minl0 / 8)
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
            if getangles:
                lines = filter_lines(img, lines)
                lines, angles = lines_kmeans(img, lines)
                print("angles: ", angles)
            else:
                angles = np.array([0,0])
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

        h_thrv = round(h_minl / 8)
        h_angl += np.pi / 14400

    if not got_hough:
        print("find_angles failed")
        exit()

    if getangles:
        drawn_lines = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR) * 0
        for line in lines:
            for x1,y1,x2,y2,r,t in line:
                cv2.line(drawn_lines,(x1,y1),(x2,y2),(0,0,250),round(2/img.sfact))
        drawn_lines = cv2.addWeighted(img.hull3ch, 0.4, drawn_lines, 0.7, 0)
        save(img, "hough_select", drawn_lines)

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

            if abs(t - tt) < 20 or abs(t - tt) > 160:
                continue

            xdiff = (l1[0][0] - l1[1][0], l2[0][0] - l2[1][0])
            ydiff = (l1[0][1] - l1[1][1], l2[0][1] - l2[1][1])

            div = determinant(xdiff, ydiff)
            if div == 0:
                j += 1
                continue

            d = (determinant(*l1), determinant(*l2))
            x = round(determinant(d, xdiff) / div)
            y = round(determinant(d, ydiff) / div)

            if img.got_hull:
                dist = cv2.pointPolygonTest(img.shull, (x, y), True)
            if img.got_hull and dist < -20:
                j += 1
                continue
            elif x > img.hwidth or y > img.hheigth or x < 0 or y < 0:
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
    img.clahe = cv2.resize(img.clahe, (img.hwidth, img.hheigth))
    img.harea = img.hwidth * img.hheigth
    return img

def update_hull(img):
    contours, _ = cv2.findContours(img.medges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cont = contours[max_index]
    hullxy = cv2.convexHull(cont)

    drawn_contours = np.empty(img.hull3ch.shape, dtype='uint8') * 0
    cv2.drawContours(drawn_contours, [hullxy], -1, (0, 255, 0), thickness=3)
    cv2.drawContours(drawn_contours, cont, -1, (255, 0, 0), thickness=3)
    drawn_contours = cv2.addWeighted(img.hull3ch, 0.4, drawn_contours, 0.7, 0)
    save(img, "updatehull", drawn_contours)
    return hullxy

def magic_lines(img):
    got_hough = False
    h_minl0 = round((img.hwidth + img.hheigth)*0.05)
    h_thrv0 = round(h_minl0 - 10)
    h_maxg0 = round(h_minl0 / 60) + 0
    h_angl0 = np.pi / 120

    tuned = 0
    if img.ext:
        maxtun = 30
        minlines = 50
        tol = 15
        print("==================extremo=================")
    else:
        maxtun = 10
        minlines = 25
        tol = 15
    h_maxg = h_maxg0
    h_minl = h_minl0
    h_thrv = h_thrv0
    h_angl = h_angl0
    j = 0
    while h_angl < (np.pi / 10):
        lines = cv2.HoughLinesP(img.canny, 1, h_angl, h_thrv,  None, h_minl, h_maxg)
        if lines is not None:
            lines = radius_theta(lines)
            lines = filter_lines(img, lines)
            lines = filter_angles(img, lines)
            if lines.shape[0] >= minlines:
                print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
                got_hough = True
                aux = np.copy(img.select_lines[:,:,0:6])
                lines = np.append(lines, aux, axis=0)
                lines = filter_angles(img, lines, tol)
                break

        if lines is not None:
            print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
        j += 1
        h_angl += np.pi / 1800
        if h_angl > (np.pi / 20) and tuned < maxtun:
            h_angl = h_angl0
            h_minl -= 5
            h_thrv -= 5
            h_maxg += 1
            tuned += 1

    if got_hough:
        drawn_lines = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR) * 0
        draw_lines = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR) * 0

        for line in lines:
            for x1,y1,x2,y2,r,t in line:
                cv2.line(draw_lines,(x1,y1),(x2,y2),(0,0,255),round(2/img.sfact))
        drawn_lines = cv2.addWeighted(img.hull3ch, 0.4, draw_lines, 0.7, 0)
        save(img, "hough_magic", drawn_lines)

        if img.got_hull:
            ko = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            img.medges = cv2.morphologyEx(img.medges, cv2.MORPH_CLOSE, ko, iterations = 1)
            img.medges = cv2.bitwise_or(img.medges, draw_lines[:,:,0])
            save(img, "medges_update", img.medges)
            img.shull = update_hull(img)
        inter = find_intersections(img, lines[:,0,:])

        drawn_circles = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR) * 0
        for p in inter:
            cv2.circle(drawn_circles, p, radius=7, color=(255, 0, 0), thickness=-1)
        drawn_circles = cv2.addWeighted(img.hull3ch, 0.4, drawn_circles, 0.7, 0)
        save(img, "intersections".format(img.basename), drawn_circles)
    else:
        print("FAILED @ {}, {}, {}, {}".format(180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
        exit()
    return lines,inter

def filter_lines(img, lines):
    """
    remove lines that are on the border of the image and are horizontal or vertical
    """
    rem = np.empty(lines.shape[0])
    rem = np.int32(rem)

    i = 0
    for line in lines:
        for x1,y1,x2,y2,r,t in line:
            if x1 < 5 and x2 < 5 or y1 < 5 and y2 < 5:
                rem[i] = 1
            elif (img.hwidth - x1) < 5 and (img.hwidth - x2) < 5 or (img.hheigth - y1) < 5 and (img.hheigth - y2) < 5:
                rem[i] = 1
            elif (x1 < 5 or (img.hwidth - x1) < 5) and (y2 < 5 or (img.hheigth - y2) < 5):
                rem[i] = 1
            elif (x2 < 5 or (img.hwidth - x2) < 5) and (y1 < 5 or (img.hheigth - y1) < 5):
                rem[i] = 1
            else:
                rem[i] = 0
        i += 1

    A = lines[rem==0]
    lines = A
    return lines

def filter_angles(img, lines, tol = 15):
    rem = np.empty(lines.shape[0])
    rem = np.int32(rem)

    i = 0
    for line in lines:
        for x1,y1,x2,y2,r,t in line:
            if abs(t - img.angles[0]) > tol and abs(t - img.angles[1]) > tol:
                if img.angles.shape[0] == 2:
                    rem[i] = 1
                elif abs(t - img.angles[2]) > tol:
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
        # plt.xticks(range(-90, 91, 10))
        # plt.hist(A[:,5], 180, [-90, 90], color = (0.9, 0.0, 0.0, 0.9))
        # plt.hist(B[:,5], 180, [-90, 90], color = (0.0, 0.0, 0.9, 0.9))
        # plt.hist(centers, 20, [-90, 90], color = (0.7, 0.7, 0.0, 0.7))
        # savefig(img, "kmeans1", fig)

    diff = []
    diff.append((abs(centers[0] - 90), -90))
    diff.append((abs(centers[0] + 90), +90))
    diff.append((abs(centers[1] - 90), -90))
    diff.append((abs(centers[1] + 90), +90))
    if centers.shape[0] > 2:
        diff.append((abs(centers[2] - 90), -90))
        diff.append((abs(centers[2] + 90), +90))

    for d,k in diff:
        if d < 20:
            if abs(centers[0] - k) > 20 and abs(centers[1] - k):
                centers = np.append(centers, k)
            elif len(centers) > 2:
                if abs(centers[2] - k) > 20:
                    centers = np.append(centers, k)
            break

    lines = np.int32(lines)
    return lines, centers

def find_corners(img, inter):
    print("inter:", inter.shape)

    psum = np.empty((inter.shape[0], 3), dtype='int32')
    psub = np.empty((inter.shape[0], 3), dtype='int32')

    psum[:,0] = inter[:,0]
    psum[:,1] = inter[:,1]
    psum[:,2] = inter[:,0] + inter[:,1]
    psub[:,0] = inter[:,0]
    psub[:,1] = inter[:,1]
    psub[:,2] = inter[:,0] - inter[:,1]

    BR = psum[np.argmax(psum[:,2])]
    BL = psub[np.argmax(psub[:,2])]
    TR = psub[np.argmin(psub[:,2])]
    TL = psum[np.argmin(psum[:,2])]

    BR = BR[0:2]
    BL = BL[0:2]
    TR = TR[0:2]
    TL = TL[0:2]

    print("points:", BR, BL, TR, TL)
    print("swapping TR and BL")
    aux = TR
    TR = BL
    BL = aux

    drawn_circles = np.copy(img.hull3ch) * 0
    # for p in BR, BL, TR, TL:                      B   G  R
    cv2.circle(drawn_circles, BR, radius=7, color=(255, 0, 0), thickness=-1)
    cv2.circle(drawn_circles, BL, radius=7, color=(0, 0, 255), thickness=-1)
    cv2.circle(drawn_circles, TR, radius=7, color=(0, 255, 0), thickness=-1)
    cv2.circle(drawn_circles, TL, radius=7, color=(255, 255, 255), thickness=-1)

    drawn_circles = cv2.addWeighted(img.hull3ch, 0.4, drawn_circles, 0.7, 0)
    save(img, "corners", drawn_circles)

    corners = np.array([BR, BL, TR, TL])

    return corners

def region(img, maxkd = 12, cmax = 12, nymax = 8, skip=False):
    c = 3
    wc = 6
    a = 0
    c0 = 12
    c5 = 0
    Amin = (c0-c5)*0.05 * img.sarea
    while c <= cmax:
        print("Amin = ", Amin)
        print("Clahe = ", c)
        alast = a
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(c, c))
        img.clahe = clahe.apply(img.wang0)
        img.wang = lwang.wang_filter(img.clahe)
        img.canny = find_canny(img, wmin=wc)
        img.medges,img.hullxy,img.got_hull,a,img.kd,img.poly,apoly,cont = find_morph(img, Amin, maxkd, skip)
        if c <= 12:
            fmedges = img.medges

        if img.got_hull:
            break
        else:
            if abs(a - alast) < img.sarea*0.02:
                if c5 <= 7:
                    c5 += 1
                print("c5 = ",c5, "NOT increasing")
                Amin = (c0-c5)*0.05 * img.sarea
            else:
                c += 1
                if wc < nymax:
                    wc += 1

    img.medges = fmedges
    return img, a, apoly, cont
